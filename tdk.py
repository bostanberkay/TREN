"""TDK dictionary lookup providers and the Turkish morphological parser used by the TDK Checker."""

import json
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cs_pipeline as cp
import reranking as mr


# ---------------------------------------------------------------------------
# Dictionary lookup providers
#
# lookup() returns a LookupResult with one of:
#   FOUND          entries is non-empty
#   NOT_FOUND      the provider understood the query and confirmed no entry
#   UNAVAILABLE    provider unusable or response unparseable; says nothing
#                  about whether the word exists
#   NETWORK_ERROR  DNS / connection / timeout / HTTP failure
# Callers pass a single term, never surrounding text. Results are cached
# in memory per provider instance, keyed by (normalized query, provider name).
# ---------------------------------------------------------------------------


STATUS_FOUND = "FOUND"
STATUS_NOT_FOUND = "NOT_FOUND"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_NETWORK_ERROR = "NETWORK_ERROR"
# Applied by the TDK Checker GUI (see mark_stale); never returned by a provider.
STATUS_STALE_RESULT = "STALE_RESULT"
ALL_STATUSES = (STATUS_FOUND, STATUS_NOT_FOUND, STATUS_UNAVAILABLE, STATUS_NETWORK_ERROR, STATUS_STALE_RESULT)

DEFAULT_TIMEOUT_SECONDS = 5.0

NOT_PROVIDED = "Not provided"


def format_field_for_display(value) -> str:
    """Display string for `value`, or NOT_PROVIDED when falsy."""
    if not value:
        return NOT_PROVIDED
    return str(value)


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

# str.lower() maps 'I'->'i' and 'İ'->'i̇' (i + combining dot); Turkish needs
# 'I'->'ı' and 'İ'->'i'. Applied before the general lower().
_TURKISH_CASE_MAP = str.maketrans({"İ": "i", "I": "ı"})


def turkish_lower(text: str) -> str:
    """Turkish-correct lowercase: 'İ'->'i', 'I'->'ı', everything else via
    the ordinary Unicode lowercase mapping. Deterministic, no locale
    dependency (never touches the process locale)."""
    if not text:
        return text
    return text.translate(_TURKISH_CASE_MAP).lower()


_APOSTROPHES_RE = re.compile(r"[’'ʼ`´]")
_EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def normalize_query(term: str) -> str:
    """NFC-normalize, strip apostrophes anywhere and edge punctuation (internal
    hyphens kept), then Turkish-lowercase. Returns "" if nothing remains."""
    if not term:
        return ""
    s = unicodedata.normalize("NFC", str(term))
    s = _APOSTROPHES_RE.sub("", s)
    s = _EDGE_PUNCT_RE.sub("", s)
    return turkish_lower(s.strip())


# TDK `gts` fields used below: madde (headword), anlamlarListe (senses, each
# with ozelliklerListe / orneklerListe), lisan, telaffuz, birlesikler, atasozu,
# deyimler. The endpoint is undocumented, so extraction is defensive and
# unrecognized data is kept in `raw`.

# TDK mixes POS tags and usage labels ("argo", "mecaz") in one ozellik list;
# this is the set treated as POS (shown verbatim), everything else is a usage label.
_KNOWN_POS_TOKENS = {"isim", "fiil", "sıfat", "zarf", "zamir", "edat", "bağlaç", "ünlem", "sayı"}


def _as_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _list_of_text(value, dict_keys: Tuple[str, ...] = ("madde", "söz", "ad")) -> Tuple[str, ...]:
    """Coerce a string (split on "|", ";" or newline), list of strings, or list
    of dicts (first non-empty key in `dict_keys`) to a tuple of non-empty
    strings; else ()."""
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(p.strip() for p in re.split(r"[|;\n]", value) if p.strip())
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    out.append(item.strip())
            elif isinstance(item, dict):
                for k in dict_keys:
                    text = _as_text(item.get(k))
                    if text:
                        out.append(text)
                        break
        return tuple(out)
    return ()


@dataclass(frozen=True)
class DictionarySense:
    definition: str
    part_of_speech: str = ""
    usage_labels: Tuple[str, ...] = ()
    examples: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "definition": self.definition,
            "part_of_speech": self.part_of_speech,
            "usage_labels": list(self.usage_labels),
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class DictionaryEntry:
    headword: str
    part_of_speech: str = ""
    origin: str = ""
    pronunciation: str = ""
    senses: Tuple[DictionarySense, ...] = ()
    compounds: Tuple[str, ...] = ()
    idioms: Tuple[str, ...] = ()
    proverbs: Tuple[str, ...] = ()
    raw: dict = field(default_factory=dict)

    @property
    def definitions(self) -> Tuple[str, ...]:
        return tuple(s.definition for s in self.senses)

    def to_dict(self) -> dict:
        return {
            "headword": self.headword,
            "part_of_speech": self.part_of_speech,
            "origin": self.origin,
            "pronunciation": self.pronunciation,
            "senses": [s.to_dict() for s in self.senses],
            "compounds": list(self.compounds),
            "idioms": list(self.idioms),
            "proverbs": list(self.proverbs),
            "raw": dict(self.raw),
        }


def _parse_tdk_entry(item: dict) -> DictionaryEntry:
    """Build a DictionaryEntry from one `gts` list element; malformed
    sub-fields degrade to empty."""
    headword = _as_text(item.get("madde"))
    origin = _as_text(item.get("lisan"))
    pronunciation = _as_text(item.get("telaffuz")) or _as_text(item.get("seslendirme"))
    compounds = _list_of_text(item.get("birlesikler"))
    idioms = _list_of_text(item.get("deyimler"), dict_keys=("madde", "söz"))
    proverbs = _list_of_text(item.get("atasozu"), dict_keys=("madde", "söz"))

    senses = []
    for s in (item.get("anlamlarListe") or []):
        if not isinstance(s, dict):
            continue
        definition = _as_text(s.get("anlam"))
        if not definition:
            continue
        pos = ""
        usage_labels = []
        for prop in (s.get("ozelliklerListe") or []):
            if not isinstance(prop, dict):
                continue
            label = _as_text(prop.get("tam_adi")) or _as_text(prop.get("ozellik_kodu"))
            if not label:
                continue
            if not pos and turkish_lower(label) in _KNOWN_POS_TOKENS:
                pos = label
            else:
                usage_labels.append(label)
        examples = tuple(
            _as_text(ex.get("ornek")) for ex in (s.get("orneklerListe") or [])
            if isinstance(ex, dict) and _as_text(ex.get("ornek"))
        )
        senses.append(DictionarySense(definition=definition, part_of_speech=pos,
                                       usage_labels=tuple(usage_labels), examples=examples))

    known_keys = {"madde", "lisan", "telaffuz", "seslendirme", "birlesikler",
                  "deyimler", "atasozu", "anlamlarListe"}
    raw_extra = {k: v for k, v in item.items() if k not in known_keys}

    entry_pos = next((s.part_of_speech for s in senses if s.part_of_speech), "")
    return DictionaryEntry(
        headword=headword, part_of_speech=entry_pos, origin=origin, pronunciation=pronunciation,
        senses=tuple(senses), compounds=compounds, idioms=idioms, proverbs=proverbs, raw=raw_extra,
    )


@dataclass(frozen=True)
class LookupResult:
    query: str
    normalized_query: str
    status: str
    source: str
    entries: Tuple[object, ...] = ()
    message: str = ""
    from_cache: bool = False

    def to_dict(self) -> dict:
        def _entry_dict(e):
            return e.to_dict() if hasattr(e, "to_dict") else dict(e)
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "status": self.status,
            "source": self.source,
            "entries": [_entry_dict(e) for e in self.entries],
            "message": self.message,
            "from_cache": self.from_cache,
        }


def mark_stale(result: "LookupResult") -> "LookupResult":
    """Copy of `result` with STATUS_STALE_RESULT, set by the TDK Checker GUI
    when root/segments are edited after a lookup."""
    return LookupResult(query=result.query, normalized_query=result.normalized_query,
                         status=STATUS_STALE_RESULT, source=result.source,
                         entries=result.entries, message=result.message,
                         from_cache=result.from_cache)


class DictionaryProvider:
    """Base provider. Subclasses implement `_do_lookup`, not `lookup`, so
    caching stays uniform. `name` keys the cache and labels results."""

    name = "base"

    def __init__(self):
        self._cache: Dict[Tuple[str, str], LookupResult] = {}

    def lookup(self, term: str) -> LookupResult:
        """Never raises; subclass failures become UNAVAILABLE."""
        normalized = normalize_query(term)
        cache_key = (normalized, self.name)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return LookupResult(
                query=term, normalized_query=normalized, status=cached.status,
                source=cached.source, entries=cached.entries, message=cached.message,
                from_cache=True,
            )
        if not normalized:
            result = LookupResult(query=term, normalized_query=normalized, status=STATUS_NOT_FOUND,
                                   source=self.name, message="empty query after normalization")
            self._cache[cache_key] = result
            return result
        try:
            result = self._do_lookup(term, normalized)
        except Exception as e:
            result = LookupResult(query=term, normalized_query=normalized, status=STATUS_UNAVAILABLE,
                                   source=self.name, message=f"provider failure: {e}")
        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        self._cache.clear()

    def _do_lookup(self, term: str, normalized: str) -> LookupResult:
        raise NotImplementedError


# lookup() is synchronous and blocking; callers run it off the UI thread.

DEFAULT_TDK_URL = "https://sozluk.gov.tr/gts"


class TDKProvider(DictionaryProvider):
    """Best-effort client for the undocumented sozluk.gov.tr `gts` endpoint.
    Unrecognized responses -> UNAVAILABLE (never a fabricated FOUND);
    HTTP/network failures -> NETWORK_ERROR."""

    name = "tdk"

    def __init__(self, base_url: str = DEFAULT_TDK_URL, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 opener: Optional[Callable[[str, float], bytes]] = None):
        super().__init__()
        self.base_url = base_url
        self.timeout = timeout
        # Injectable (url, timeout) -> bytes; tests use it to simulate network failures.
        self._opener = opener or self._http_get

    @staticmethod
    def _http_get(url: str, timeout: float) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "TREN-TDK-Checker/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _do_lookup(self, term: str, normalized: str) -> LookupResult:
        # Percent-encodes non-ASCII (ç/ğ/ı/ö/ş/ü) as UTF-8.
        url = f"{self.base_url}?ara={urllib.parse.quote(normalized, safe='')}"
        try:
            raw = self._opener(url, self.timeout)
        except urllib.error.HTTPError as e:
            return LookupResult(query=term, normalized_query=normalized, status=STATUS_NETWORK_ERROR,
                                 source=self.name, message=f"HTTP error {e.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return LookupResult(query=term, normalized_query=normalized, status=STATUS_NETWORK_ERROR,
                                 source=self.name, message=f"network error: {e}")

        return self._parse_response(term, normalized, raw)

    @staticmethod
    def _parse_response(term: str, normalized: str, raw: bytes) -> LookupResult:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            return LookupResult(query=term, normalized_query=normalized, status=STATUS_UNAVAILABLE,
                                 source=TDKProvider.name, message=f"non-UTF-8 response: {e}")
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            return LookupResult(query=term, normalized_query=normalized, status=STATUS_UNAVAILABLE,
                                 source=TDKProvider.name, message=f"malformed (non-JSON) response: {e}")

        # Observed (undocumented) shapes: found = non-empty list of dicts with
        # "madde"; not found = {"error": ...} or []. Anything else is UNAVAILABLE.
        # NOT_FOUND carries our own fixed message, never TDK's Turkish error text.
        if isinstance(data, dict) and "error" in data:
            return LookupResult(query=term, normalized_query=normalized, status=STATUS_NOT_FOUND,
                                 source=TDKProvider.name, message="no dictionary entry found")
        if isinstance(data, list):
            if not data:
                return LookupResult(query=term, normalized_query=normalized, status=STATUS_NOT_FOUND,
                                     source=TDKProvider.name, message="no dictionary entry found")
            if all(isinstance(item, dict) and "madde" in item for item in data):
                entries = tuple(_parse_tdk_entry(item) for item in data)
                return LookupResult(query=term, normalized_query=normalized, status=STATUS_FOUND,
                                     source=TDKProvider.name, entries=entries)

        return LookupResult(query=term, normalized_query=normalized, status=STATUS_UNAVAILABLE,
                             source=TDKProvider.name,
                             message="unrecognized response structure (TDK endpoint may have changed)")


class UnavailableProvider(DictionaryProvider):
    name = "offline"

    def _do_lookup(self, term: str, normalized: str) -> LookupResult:
        return LookupResult(query=term, normalized_query=normalized, status=STATUS_UNAVAILABLE,
                             source=self.name, message="dictionary lookup is offline")


class MockDictionaryProvider(DictionaryProvider):
    """In-memory provider for tests. `responses` maps a normalized query to a
    status or (status, entries); unmatched queries get `default_status`.
    `delay_seconds` sleeps before returning."""

    name = "mock"

    def __init__(self, responses: Optional[Dict[str, object]] = None,
                 default_status: str = STATUS_NOT_FOUND, delay_seconds: float = 0.0):
        super().__init__()
        self.responses = dict(responses or {})
        self.default_status = default_status
        self.delay_seconds = delay_seconds
        self.call_count = 0
        self.calls = []

    def _do_lookup(self, term: str, normalized: str) -> LookupResult:
        self.call_count += 1
        self.calls.append(term)
        if self.delay_seconds:
            import time
            time.sleep(self.delay_seconds)
        spec = self.responses.get(normalized)
        if spec is None:
            status, entries = self.default_status, ()
        elif isinstance(spec, tuple):
            status, entries = spec
        else:
            status, entries = spec, ()
        if status not in ALL_STATUSES:
            raise ValueError(f"MockDictionaryProvider: unknown status {status!r}")
        return LookupResult(query=term, normalized_query=normalized, status=status,
                             source=self.name, entries=tuple(entries))


# ---------------------------------------------------------------------------
# Turkish morphological parser
#
# Proposes a root, ordered suffix segments, a category and per-boundary
# explanations for user review; never assigns a language label.
#
# Candidates: reranking's nominal/verbal enumeration, this module's atomic
# VERB_TAM/VERB_PERSON tables, and the unsplit token. All compete in one
# additive score (_score_candidate). There is deliberately no "whole token
# in lexicon -> never split" rule: the frequency lexicons contain inflected
# surface forms ("geldi", "filmin"), so membership alone cannot decide.
# Score terms: stem lexicon tier/rank, structural bonus for an atomic
# TAM(+person) match or all-valid nominal segments, longer stem, penalty per
# single-character segment, soft vowel-harmony agreement. Ties break on
# stem length. Not a calibrated model.
# ---------------------------------------------------------------------------


MIN_ROOT_LEN = 2

# reranking's non-atomic verbal fallback proposes coincidental short stems
# ("ka"+"le"+"m" for "kalem"); trust it only for stems of 3+ characters.
MR_VERBAL_MIN_ROOT_LEN = 3

FRONT_VOWELS = set("eiöü")
BACK_VOWELS = set("aıou")
ROUNDED_VOWELS = set("oöuü")
UNROUNDED_VOWELS = set("aeıi")
ALL_VOWELS = FRONT_VOWELS | BACK_VOWELS


def _last_vowel(s: str) -> Optional[str]:
    return next((c for c in reversed(s) if c in ALL_VOWELS), None)


def _first_vowel(s: str) -> Optional[str]:
    return next((c for c in s if c in ALL_VOWELS), None)


def vowel_harmony_consistent(stem: str, first_segment: str) -> Optional[bool]:
    """Front/back agreement between the stem's last vowel and the segment's
    first vowel; None when either side has no vowel."""
    sv, fv = _last_vowel(stem), _first_vowel(first_segment)
    if sv is None or fv is None:
        return None
    return (sv in FRONT_VOWELS) == (fv in FRONT_VOWELS)


# Atomic verb TAM/person suffix entries with harmony variants; additional to reranking's tables.

@dataclass(frozen=True)
class SuffixEntry:
    underlying_form: str
    category: str
    part_of_speech: str
    surface_forms: Tuple[str, ...]
    requires_vowel_harmony: bool
    allows_consonant_alternation: bool
    valid_predecessor_categories: Tuple[str, ...]
    gloss: str


VERB_TAM_ENTRIES: Tuple[SuffixEntry, ...] = (
    SuffixEntry("-DI", "past_definite", "verb",
                ("dı", "di", "du", "dü", "tı", "ti", "tu", "tü"),
                True, True, ("root",), "past tense (-DI)"),
    SuffixEntry("-mIş", "evidential_past", "verb",
                ("mış", "miş", "muş", "müş"),
                True, False, ("root",), "evidential/reported past tense (-mIş)"),
    SuffixEntry("-Iyor", "progressive", "verb",
                ("ıyor", "iyor", "uyor", "üyor"),
                True, False, ("root",), "present progressive (-Iyor)"),
    SuffixEntry("-(y)AcAk", "future", "verb",
                ("acak", "ecek", "yacak", "yecek"),
                True, False, ("root",), "future tense (-(y)AcAk)"),
)

VERB_PERSON_ENTRIES: Tuple[SuffixEntry, ...] = (
    SuffixEntry("-m", "first_singular", "verb", ("m",), False, False,
                ("past_definite", "evidential_past"), "1st person singular agreement"),
    SuffixEntry("-n", "second_singular", "verb", ("n",), False, False,
                ("past_definite", "evidential_past"), "2nd person singular agreement"),
    SuffixEntry("-k", "first_plural", "verb", ("k",), False, False,
                ("past_definite", "evidential_past"), "1st person plural agreement"),
    SuffixEntry("-(s)InIz", "second_plural", "verb",
                ("sınız", "siniz", "sunuz", "sünüz"), True, False,
                ("past_definite", "evidential_past", "progressive", "future"), "2nd person plural agreement"),
    SuffixEntry("-lAr", "third_plural", "verb", ("lar", "ler"), True, False,
                ("past_definite", "evidential_past", "progressive", "future"), "3rd person plural agreement"),
)
# 3sg is zero-marked: deliberately no entry (a bare TAM suffix is already the complete 3sg form).


def _enumerate_structured_verb_candidates(token_l: str) -> List[Tuple[str, Tuple[str, ...], Tuple[SuffixEntry, ...]]]:
    """(stem, segments, entries) for an atomic TAM form, optionally followed by
    an allowed person form, consuming the whole remainder."""
    out = []
    n = len(token_l)
    for split in range(MIN_ROOT_LEN, n):
        stem, rest = token_l[:split], token_l[split:]
        for tam in VERB_TAM_ENTRIES:
            for form in tam.surface_forms:
                if rest == form:
                    out.append((stem, (form,), (tam,)))
                elif rest.startswith(form):
                    remainder = rest[len(form):]
                    for person in VERB_PERSON_ENTRIES:
                        if tam.category not in person.valid_predecessor_categories:
                            continue
                        if remainder in person.surface_forms:
                            out.append((stem, (form, remainder), (tam, person)))
    return out


# Read-only reuse of cs_pipeline's suffix tables to label nominal segments.

def _classify_nominal_segment(segment: str) -> Tuple[str, str]:
    seg_l = segment.lower()
    if seg_l in cp.BUFFER_N_ACC:
        return cp.BUFFER_N_ACC[seg_l], "buffer-n accusative allomorph"
    if seg_l in cp.BUFFER_N_DAT:
        return cp.BUFFER_N_DAT[seg_l], "buffer-n dative allomorph"
    if seg_l in cp.CASE_ENDINGS:
        return cp.CASE_ENDINGS[seg_l], "Turkish case suffix"
    if seg_l in cp.POSS_LONG:
        return "+".join(cp.POSS_LONG[seg_l]), "Turkish possessive suffix (plural possessor)"
    if seg_l in cp.POSS_SHORT:
        return "+".join(cp.POSS_SHORT[seg_l]), "Turkish possessive suffix"
    if seg_l in cp.PLUR:
        return cp.PLUR[seg_l], "Turkish plural suffix"
    if seg_l in cp.DERIV_SUFFIXES:
        return "+".join(cp.DERIV_SUFFIXES[seg_l]), "Turkish derivational suffix"
    if seg_l in ("ı", "i", "u", "ü"):
        return "Amb=P3sg_or_Acc", "ambiguous 3rd-person-possessive/accusative suffix"
    return "unrecognized", "unrecognized segment"


@dataclass(frozen=True)
class SegmentExplanation:
    segment: str
    category: str
    valid: bool
    rule: str

    def to_dict(self) -> dict:
        return {"segment": self.segment, "category": self.category, "valid": self.valid, "rule": self.rule}


# If the winning split beats a lexicon-attested unsplit reading by less than
# this margin, the readings are too close to call from frequency alone
# ("kalem" vs "kale"+"m", margin ~30; correct splits seen at >=140, e.g. "kitaplarda").
AMBIGUITY_MARGIN = 50.0

CATEGORY_FULL_LEXICAL = "full_turkish_lexical_item"
CATEGORY_ENGLISH_ROOT = "english_root_turkish_suffix"
CATEGORY_AMBIGUOUS = "ambiguous_candidate"
CATEGORY_INVALID = "invalid_parser_proposal"
CATEGORY_MANUAL = "manual_correction"

_POS_VERB = "verb"
_POS_NOUN = "noun"
_POS_UNKNOWN = "unknown"


@dataclass(frozen=True)
class _Candidate:
    stem: str
    segments: Tuple[str, ...]
    source: str  # "verb_structured" | "verbal" | "nominal" | "full_token"
    stem_in_top: bool
    stem_in_all: bool
    stem_in_english: bool
    harmony: Optional[bool]
    explanations: Tuple[SegmentExplanation, ...]
    all_segments_valid: bool = False


def _score_candidate(c: _Candidate) -> Tuple[float, int]:
    score = 0.0
    if c.stem_in_top:
        score += 1000.0
    elif c.stem_in_all:
        score += 500.0
    if c.stem_in_english and not c.stem_in_all and not c.stem_in_top:
        score += 400.0
    if c.source == "verb_structured":
        score += 300.0
    elif c.source == "verbal":
        score += 200.0
    elif c.source == "nominal":
        # Per-segment credit so an attested inflected form ("kitaplar") cannot
        # outscore "kitap"+"lar"+"da" merely by being longer.
        score += sum(90.0 for e in c.explanations if e.valid)
    score += len(c.stem) * 10.0
    score -= sum(1 for s in c.segments if len(s) == 1) * 50.0
    if c.harmony is True:
        score += 10.0
    elif c.harmony is False:
        score -= 5.0
    # Returned separately so ties can be broken deterministically.
    return score, len(c.stem)


def _lexicon_flags(stem: str, annotator) -> Tuple[bool, bool, bool]:
    stem_l = stem.lower()
    top = stem_l in getattr(annotator, "turkish_freq_top", ())
    allw = top or stem_l in getattr(annotator, "turkish_freq_all", ())
    eng = stem_l in getattr(annotator, "english_freq_words", ())
    return top, allw, eng


def _build_candidates(token: str, annotator) -> List[_Candidate]:
    token_l = token.lower()
    candidates: List[_Candidate] = []

    for stem, segments, entries in _enumerate_structured_verb_candidates(token_l):
        if len(stem) < MIN_ROOT_LEN:
            continue
        top, allw, eng = _lexicon_flags(stem, annotator)
        harmony = vowel_harmony_consistent(stem, segments[0]) if segments else None
        explanations = tuple(
            SegmentExplanation(seg, entry.category, True,
                                f"structured Turkish verb suffix ({entry.underlying_form}): {entry.gloss}")
            for seg, entry in zip(segments, entries)
        )
        candidates.append(_Candidate(stem, segments, "verb_structured", top, allw, eng, harmony,
                                      explanations, all_segments_valid=True))

    try:
        mr_candidates = mr.enumerate_candidate_analyses(token, annotator, verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4E)
    except Exception:
        mr_candidates = []
    for cand in mr_candidates:
        stem = cand.stem
        min_len = MR_VERBAL_MIN_ROOT_LEN if cand.source == "verbal" else MIN_ROOT_LEN
        if len(stem) < min_len:
            continue
        top, allw, eng = _lexicon_flags(stem, annotator)
        segments = tuple(cand.segments)
        harmony = vowel_harmony_consistent(stem, segments[0]) if segments else None
        if cand.source == "verbal":
            explanations = tuple(
                SegmentExplanation(seg, "Verbal", True,
                                    "Turkish verbal suffix (experimental morphology table)")
                for seg in segments
            )
            all_valid = True
        else:
            explanations = _explanations_for_nominal(segments)
            all_valid = bool(segments) and all(e.valid for e in explanations)
        candidates.append(_Candidate(stem, segments, cand.source, top, allw, eng, harmony,
                                      explanations, all_segments_valid=all_valid))

    # The unsplit whole token competes as one more scored candidate.
    top, allw, eng = _lexicon_flags(token, annotator)
    candidates.append(_Candidate(token, (), "full_token", top, allw, eng, None, ()))

    return candidates


def _explanations_for_nominal(segments: Tuple[str, ...]) -> Tuple[SegmentExplanation, ...]:
    out = []
    for seg in segments:
        category, rule = _classify_nominal_segment(seg)
        out.append(SegmentExplanation(seg, category, category != "unrecognized", rule))
    return tuple(out)


def _select_best(candidates: List[_Candidate]) -> Optional[_Candidate]:
    if not candidates:
        return None
    scored = [(_score_candidate(c), c) for c in candidates]
    scored.sort(key=lambda pair: (pair[0][0], pair[0][1]), reverse=True)
    return scored[0][1]


def _categorize(candidate: Optional[_Candidate]) -> Tuple[str, str]:
    """Returns (category, part_of_speech) for the winning candidate."""
    if candidate is None:
        return CATEGORY_INVALID, _POS_UNKNOWN
    if candidate.source in ("verb_structured", "verbal"):
        pos = _POS_VERB
    elif candidate.source == "nominal":
        pos = _POS_NOUN
    else:
        pos = _POS_UNKNOWN
    if candidate.stem_in_top or candidate.stem_in_all:
        return CATEGORY_FULL_LEXICAL, pos
    if candidate.stem_in_english:
        return CATEGORY_ENGLISH_ROOT, pos
    if candidate.segments:
        return CATEGORY_AMBIGUOUS, pos
    return CATEGORY_INVALID, pos


@dataclass(frozen=True)
class ParseResult:
    token: str
    root: str
    segments: Tuple[str, ...]
    success: bool
    source: str  # "full_lexical_item" | "verb_structured" | "verbal" | "nominal" | "manual" | "whole_token_fallback"
    category: str  # CATEGORY_* constant
    part_of_speech: str  # "verb" | "noun" | "unknown"
    reason: str
    segment_explanations: Tuple[SegmentExplanation, ...] = ()
    harmony_consistent: Optional[bool] = None
    stem_in_turkish_lexicon: bool = False
    stem_in_english_lexicon: bool = False

    @property
    def suffix(self) -> str:
        return "".join(self.segments)

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "root": self.root,
            "segments": list(self.segments),
            "success": self.success,
            "source": self.source,
            "category": self.category,
            "part_of_speech": self.part_of_speech,
            "reason": self.reason,
            "segment_explanations": [e.to_dict() for e in self.segment_explanations],
            "harmony_consistent": self.harmony_consistent,
            "stem_in_turkish_lexicon": self.stem_in_turkish_lexicon,
            "stem_in_english_lexicon": self.stem_in_english_lexicon,
        }


def _whole_token_fallback(token: str, reason: str) -> ParseResult:
    return ParseResult(token=token, root=token, segments=(), success=False,
                        source="whole_token_fallback", category=CATEGORY_INVALID,
                        part_of_speech=_POS_UNKNOWN, reason=reason)


def parse_token(token: str, annotator) -> ParseResult:
    """Hierarchical parse of `token`. Never raises (failures degrade to a
    whole-token fallback) and never assigns a language label."""
    if not token or not str(token).strip():
        return _whole_token_fallback(token or "", "empty token")

    token_l = token.lower()

    try:
        candidates = _build_candidates(token, annotator)
    except Exception as e:
        return _whole_token_fallback(token, f"parser error: {e}")

    # _build_candidates always appends a "whole token, unsplit" candidate,
    # so this pool is never empty and `best` is never None here.
    best = _select_best(candidates)

    reconstructed = f"{best.stem}{''.join(best.segments)}"
    if reconstructed.lower() != token_l:
        return _whole_token_fallback(token, "malformed segmentation: stem+suffix does not reconstruct token")

    category, pos = _categorize(best)
    lexicon_confirmed = best.stem_in_top or best.stem_in_all or best.stem_in_english

    if not best.segments and not lexicon_confirmed:
        # Unsplit winner not confirmed by any lexicon: unanalyzable, not a confident result.
        return _whole_token_fallback(token, "no valid Turkish suffix-chain split found")

    source = "full_lexical_item" if (best.source == "full_token" and not best.segments) else best.source
    n_segments = len(best.segments)
    ambiguous_margin = False
    if best.segments and category in (CATEGORY_FULL_LEXICAL, CATEGORY_ENGLISH_ROOT):
        full_tok = next((c for c in candidates if c.source == "full_token"), None)
        if full_tok is not None and (full_tok.stem_in_top or full_tok.stem_in_all or full_tok.stem_in_english):
            gap = _score_candidate(best)[0] - _score_candidate(full_tok)[0]
            if gap < AMBIGUITY_MARGIN:
                ambiguous_margin = True
                category = CATEGORY_AMBIGUOUS

    if ambiguous_margin:
        reason = (
            f"{n_segments} suffix segment(s) found via "
            f"{'structured verb morphology' if best.source == 'verb_structured' else best.source + ' analysis'}"
            f", but the unsplit token \"{token}\" is itself also a plausible lexical item -- "
            "ambiguous without a dictionary lookup"
        )
    elif n_segments:
        reason = (
            f"{n_segments} suffix segment(s) found via "
            f"{'structured verb morphology' if best.source == 'verb_structured' else best.source + ' analysis'}"
        )
    else:
        reason = "the complete token is itself a known Turkish lexical item; no suffix split proposed"
    return ParseResult(
        token=token, root=best.stem, segments=best.segments, success=True, source=source,
        category=category, part_of_speech=pos, reason=reason,
        segment_explanations=best.explanations, harmony_consistent=best.harmony,
        stem_in_turkish_lexicon=(best.stem_in_top or best.stem_in_all),
        stem_in_english_lexicon=best.stem_in_english,
    )


# Manual correction: always category MANUAL; `success` only reflects whether
# root+segments reconstruct the token, never agreement with the automatic parser.

_SPLIT_RE = re.compile(r"[+\-\s]+")


def reparse_with_manual_root(token: str, root: str) -> ParseResult:
    if not token:
        return ParseResult(token=token or "", root=root or "", segments=(), success=False,
                            source="manual", category=CATEGORY_INVALID, part_of_speech=_POS_UNKNOWN,
                            reason="empty token")
    if not root or not token.lower().startswith(root.lower()):
        return ParseResult(token=token, root=token, segments=(), success=False,
                            source="manual", category=CATEGORY_INVALID, part_of_speech=_POS_UNKNOWN,
                            reason="root is not a prefix of the token")
    remainder = token[len(root):]
    segments = (remainder,) if remainder else ()
    return ParseResult(token=token, root=root, segments=segments, success=True,
                        source="manual", category=CATEGORY_MANUAL, part_of_speech=_POS_UNKNOWN,
                        reason="manually corrected")


def segments_from_text(token: str, root: str, segments_text: str) -> ParseResult:
    root = (root or "").strip()
    parts = [p.strip() for p in _SPLIT_RE.split(segments_text or "") if p.strip()]
    segments = tuple(parts)
    reconstructed = f"{root}{''.join(segments)}"
    ok = bool(root) and reconstructed.lower() == (token or "").lower()
    return ParseResult(
        token=token or "", root=root, segments=segments, success=ok,
        source="manual", category=(CATEGORY_MANUAL if ok else CATEGORY_INVALID), part_of_speech=_POS_UNKNOWN,
        reason="manually corrected" if ok else "root + segments do not reconstruct the token",
    )


# Reads only the word lists, avoiding a full Annotator (fastText/Stanza) load.

def load_lexicon_annotator(freq_tr: str = "frequent_tr_words.txt",
                            freq_en: str = "frequent_en_words.txt"):
    from cs_pipeline import Annotator
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = set()
    obj.turkish_freq_all = set()
    obj.english_freq_words = set()
    obj.ner = None
    try:
        with open(freq_tr, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    continue
                w = parts[0].lower()
                obj.turkish_freq_all.add(w)
                if i < 1000:
                    obj.turkish_freq_top.add(w)
    except OSError:
        pass
    try:
        with open(freq_en, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    obj.english_freq_words.add(parts[0].lower())
    except OSError:
        pass
    return obj

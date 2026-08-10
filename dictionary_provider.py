# dictionary_provider.py
"""Dictionary lookup provider abstraction for the TDK Checker tool.

Purpose
--------------------------------------------------------------------------
The TDK Checker (cs_annotator_app.py) never talks to a network endpoint
directly -- it only ever calls `DictionaryProvider.lookup(term)`. This
keeps the GUI fully decoupled from *how* (or whether) a lookup happens, so
it can run against a real online provider, a deterministic mock (all
tests), or an always-offline provider with the exact same code path.

Why not "the official TDK API"
--------------------------------------------------------------------------
The Turkish Language Association (TDK) does not publish an official,
documented public API. `sozluk.gov.tr`'s `gts` JSON endpoint is widely
used by third-party tools, but it is undocumented, unversioned, and can
change its response shape or disappear at any time without notice. This
module never assumes that endpoint is stable or "official" -- every
response is parsed defensively, and ANY unexpected shape, HTTP error, or
network failure degrades to `UNAVAILABLE`/`NETWORK_ERROR` rather than
raising or fabricating a result. See TDKProvider's docstring and the
project report's "Limitations" section for the practical consequences of
this (occasional false NOT_FOUND/UNAVAILABLE if TDK changes their
response shape -- never a fabricated FOUND).

Status contract
--------------------------------------------------------------------------
Every `lookup()` call returns a `LookupResult` with exactly one of four
statuses:
  FOUND         -- the term (or a close variant) was found, `entries` is
                   non-empty.
  NOT_FOUND     -- the provider was reachable and understood, and
                   confirmed the term has no entry.
  UNAVAILABLE   -- the provider could not be used at all (offline
                   provider, or a reachable-but-unparseable response --
                   e.g. TDK changed its response shape). Not a claim
                   about whether the word exists.
  NETWORK_ERROR -- a genuine network-level failure (DNS, connection
                   refused, timeout, malformed HTTP). Distinct from
                   UNAVAILABLE so a user can tell "can't reach the
                   internet" from "reached it, but the response made no
                   sense" -- both are equally "no answer", but the
                   distinction matters for diagnosing which one is worth
                   retrying.

Privacy
--------------------------------------------------------------------------
`lookup()` takes a single `term` (a token, a root/lemma, or a single
morphological segment) -- never a sentence, never surrounding context.
Callers (cs_annotator_app.py) are responsible for only ever passing the
one string the user explicitly asked to check; this module has no way to
enforce that from here, but it is designed around exactly that contract
(one term in, one term looked up, nothing else transmitted).

Caching
--------------------------------------------------------------------------
Every provider caches results in-memory, keyed by
(normalized_query, provider.name) -- looking up the same term twice
within one session never repeats the network round-trip. The cache is
per-provider-instance and never persisted to disk or shared globally.
"""

import json
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

STATUS_FOUND = "FOUND"
STATUS_NOT_FOUND = "NOT_FOUND"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_NETWORK_ERROR = "NETWORK_ERROR"
# Not returned by any provider's lookup() call -- a provider always answers
# a fresh query with one of the four statuses above. STALE_RESULT is
# applied by the CALLER (cs_annotator_app.py's TDK Checker window) to a
# previously-fetched LookupResult that no longer corresponds to the
# current root/segments the user is editing -- see mark_stale() below.
STATUS_STALE_RESULT = "STALE_RESULT"
ALL_STATUSES = (STATUS_FOUND, STATUS_NOT_FOUND, STATUS_UNAVAILABLE, STATUS_NETWORK_ERROR, STATUS_STALE_RESULT)

DEFAULT_TIMEOUT_SECONDS = 5.0

# Used by the GUI layer wherever a field extracted from a dictionary entry
# is absent -- never a guessed or blank value, always this exact string.
NOT_PROVIDED = "Not provided"


def format_field_for_display(value) -> str:
    """`value` -> a display string, substituting NOT_PROVIDED for anything
    falsy (None, "", 0, empty container). Never guesses a value -- this is
    purely a presentation-layer fallback for genuinely absent data."""
    if not value:
        return NOT_PROVIDED
    return str(value)


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

# Turkish-correct case folding: Python's default str.lower()/casefold() is
# not Turkish-locale-aware -- 'I'.lower() == 'i' and 'İ'.lower() == 'i̇'
# (i + combining dot above) in the default (non-Turkish) Unicode casing,
# which is wrong for Turkish text ('I' should fold to dotless 'ı', 'İ' to
# dotted 'i'). Applied as an explicit translation BEFORE the general
# .lower() call so every other character still folds normally.
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
    """Deterministic normalization for TDK lookup/caching:
    - Unicode NFC normalization (so visually-identical strings with
      different combining-character decompositions compare equal).
    - Apostrophes (straight, curly, backtick, acute -- anything a user's
      keyboard/autocorrect might produce) removed anywhere in the string,
      not just at the edges, since TREN tokens carry them internally
      (e.g. "kitab'ın").
    - Leading/trailing punctuation and whitespace (including hyphens)
      stripped -- mirrors annotation_model.freq_normalize_token's
      edge-punctuation convention, but internal hyphens (genuine compound
      words) are preserved.
    - Turkish-correct lowercasing (see turkish_lower).
    Returns "" for a token with no normalizable content (never raises).
    """
    if not term:
        return ""
    s = unicodedata.normalize("NFC", str(term))
    s = _APOSTROPHES_RE.sub("", s)
    s = _EDGE_PUNCT_RE.sub("", s)
    return turkish_lower(s.strip())


# ---------------------------------------------------------------------------
# Rich dictionary entry model
# ---------------------------------------------------------------------------
# TDK's undocumented `gts` endpoint (see module docstring) returns, per
# entry, roughly: "madde" (headword), "anlamlarListe" (senses, each with an
# "anlam" definition text and its own "ozelliklerListe" of POS/usage-label
# properties and "orneklerListe" of example sentences), "lisan" (origin/
# etymology), "telaffuz" (pronunciation), "birlesikler" (compound/related
# forms), "atasozu" (proverbs), "deyimler" (idioms). None of this is
# guaranteed stable -- every extraction below is defensive (missing/
# malformed fields degrade to empty, never raise), and anything not
# recognized is preserved in `raw` rather than silently dropped, so a
# future GUI panel can still surface it structurally without ever having
# to dump raw HTML/JSON into the UI.

# TDK's ozellik ("property") labels mix true part-of-speech tags with
# register/usage labels (e.g. "argo" slang, "mecaz" figurative) in the same
# undocumented list, with no reliable machine-readable distinction between
# the two. This is the smallest defensible set of tokens actually used as
# POS tags by TDK (surfaced verbatim, never translated -- see module/task
# requirement to show POS "as given by TDK: isim, fiil, sıfat, zarf,
# zamir"); anything else in a sense's ozelliklerListe is treated as a
# usage label instead.
_KNOWN_POS_TOKENS = {"isim", "fiil", "sıfat", "zarf", "zamir", "edat", "bağlaç", "ünlem", "sayı"}


def _as_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _list_of_text(value, dict_keys: Tuple[str, ...] = ("madde", "söz", "ad")) -> Tuple[str, ...]:
    """Defensively coerces a TDK field that might be a "|"-joined string, a
    list of strings, or a list of dicts (checked against `dict_keys` in
    order) into a flat tuple of non-empty strings. Anything else -> ()."""
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
    """Defensively builds one DictionaryEntry from one element of TDK's
    `gts` response list. Never raises -- any malformed sub-field is simply
    skipped, degrading that one field to empty rather than failing the
    whole entry."""
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


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

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
    """Returns a copy of `result` with status STATUS_STALE_RESULT, entries/
    message preserved for display ("here is what we found, but it no
    longer corresponds to your current edits"). Used exclusively by the
    TDK Checker GUI when the user edits root/segments after a lookup has
    already completed -- never produced by a provider's lookup() call
    itself."""
    return LookupResult(query=result.query, normalized_query=result.normalized_query,
                         status=STATUS_STALE_RESULT, source=result.source,
                         entries=result.entries, message=result.message,
                         from_cache=result.from_cache)


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class DictionaryProvider:
    """Base class / interface. `name` identifies the provider for caching
    and for display in the UI's source/status field. Subclasses must
    implement `_do_lookup`, not override `lookup` directly, so caching
    stays consistent across every provider."""

    name = "base"

    def __init__(self):
        self._cache: Dict[Tuple[str, str], LookupResult] = {}

    def lookup(self, term: str) -> LookupResult:
        """Never raises: any subclass failure is caught here and reported
        as UNAVAILABLE, so a provider bug can never crash the GUI."""
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


# ---------------------------------------------------------------------------
# Real TDK provider (best-effort, undocumented endpoint -- see module
# docstring). Off-thread execution is the CALLER's responsibility
# (cs_annotator_app.py) -- lookup() itself is a plain, synchronous,
# blocking call, deliberately, so it stays simple to test in isolation.
# ---------------------------------------------------------------------------

DEFAULT_TDK_URL = "https://sozluk.gov.tr/gts"


class TDKProvider(DictionaryProvider):
    """Best-effort provider for the widely-used (but undocumented, not
    officially published) sozluk.gov.tr `gts` JSON endpoint. Every
    assumption about that endpoint's response shape is defensive: if TDK
    changes it, or returns something this provider doesn't recognize, the
    result is UNAVAILABLE -- never a crash, never a fabricated FOUND.

    A genuine network-level failure (DNS, connection refused, timeout,
    non-2xx HTTP status) is reported as NETWORK_ERROR instead, so a user
    can distinguish "no internet" from "reached TDK, but couldn't parse
    what came back."
    """

    name = "tdk"

    def __init__(self, base_url: str = DEFAULT_TDK_URL, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 opener: Optional[Callable[[str, float], bytes]] = None):
        super().__init__()
        self.base_url = base_url
        self.timeout = timeout
        # `opener` is an injectable (url, timeout) -> bytes function, used
        # by tests to simulate network conditions (timeout, malformed
        # bytes, ...) without ever making a real request. Production uses
        # the default urllib-based `_http_get`.
        self._opener = opener or self._http_get

    @staticmethod
    def _http_get(url: str, timeout: float) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "TREN-TDK-Checker/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _do_lookup(self, term: str, normalized: str) -> LookupResult:
        # URL-encode correctly, including Turkish characters -- quote()
        # with the default UTF-8 encoding percent-encodes every non-ASCII
        # byte, which is exactly right for Turkish (ç/ğ/ı/ö/ş/ü etc.).
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

        # Known TDK gts shapes (as observed, NOT documented/guaranteed):
        #   found:     a non-empty list of dicts, each with a "madde" key
        #              (headword) and usually "anlamlarListe" (senses).
        #   not found: {"error": "Sonuç bulunamadı"} (a dict, not a list)
        #              or an empty list.
        # Anything else (changed structure, a string, null, a list of
        # non-dicts, ...) is explicitly NOT assumed to mean either FOUND
        # or NOT_FOUND -- it's UNAVAILABLE, per the task requirement to
        # treat API/HTML changes as a provider failure.
        #
        # NOT_FOUND's message is always this module's own fixed English
        # text, NEVER the raw upstream string (TDK's own "error" value is
        # Turkish system text like "Sonuç bulunamadı") -- the GUI must
        # never surface untranslated backend text to the user.
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


# ---------------------------------------------------------------------------
# Always-offline provider -- never touches the network at all. Used when
# the application is deliberately run offline, and as the safe default
# object before any lookup is explicitly requested.
# ---------------------------------------------------------------------------

class UnavailableProvider(DictionaryProvider):
    name = "offline"

    def _do_lookup(self, term: str, normalized: str) -> LookupResult:
        return LookupResult(query=term, normalized_query=normalized, status=STATUS_UNAVAILABLE,
                             source=self.name, message="dictionary lookup is offline")


# ---------------------------------------------------------------------------
# Deterministic mock provider -- the ONLY provider any test may use.
# ---------------------------------------------------------------------------

class MockDictionaryProvider(DictionaryProvider):
    """Deterministic, in-memory provider for tests. `responses` maps a
    NORMALIZED query string to either a status string (entries default
    empty) or a (status, entries) tuple. Any query not in `responses`
    returns `default_status` (default NOT_FOUND). `delay_seconds`, if
    set, sleeps synchronously before returning -- lets tests exercise the
    "stale response" / "no GUI freeze" scenarios deterministically without
    any real network dependency."""

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

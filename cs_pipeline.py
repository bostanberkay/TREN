# cs_pipeline.py


import re
import fasttext
import stanza
from functools import lru_cache

# Config
DEFAULTS = {
    "FEATURE_LANGUAGE_PER_ITEM": True,
    "FEATURE_MATRIX_LANGUAGE":   True,
    "FEATURE_EMBEDDED_LANGUAGE": True,
    "FEATURE_SENTENCE_ID":       True,
    "NER_ENABLED":               True,
    "FT_EN_MIN":                 0.80,
    "FT_TR_MIN":                 0.80,
    "MIXED_STRICT":              True,
    "EMIT_MIXED_SUFFIX":         False,
    "MIXED_TR_WEIGHT":           0.6,
    "MIXED_EN_WEIGHT":           0.4,
    "EMBED_MIN_COUNT":           1,
    "EMBED_MIN_RATIO":           0.20,
}

# Regex
URL_RE     = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NUMERIC_RE = re.compile(r"^\d+([.,:/-]\d+)*$")
EMOJI_RE   = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

EN_CONTRACTIONS = {"s", "re", "ve", "m", "ll", "d", "t"}

DERIV_SUFFIXES = {
    "lık": ("Deriv=LIK", "DerivPOS=NOUN"),
    "lik": ("Deriv=LIK", "DerivPOS=NOUN"),
    "luk": ("Deriv=LIK", "DerivPOS=NOUN"),
    "lük": ("Deriv=LIK", "DerivPOS=NOUN"),
    "sal": ("Deriv=SAL", "DerivPOS=ADJ"),
    "sel": ("Deriv=SAL", "DerivPOS=ADJ"),
    "li":  ("Deriv=LI",  "DerivPOS=ADJ"),
    "lı":  ("Deriv=LI",  "DerivPOS=ADJ"),
    "lu":  ("Deriv=LI",  "DerivPOS=ADJ"),
    "lü":  ("Deriv=LI",  "DerivPOS=ADJ"),
    "siz": ("Deriv=SIZ", "DerivPOS=ADJ"),
    "suz": ("Deriv=SIZ", "DerivPOS=ADJ"),
    "süz": ("Deriv=SIZ", "DerivPOS=ADJ"),
    "sız": ("Deriv=SIZ", "DerivPOS=ADJ"),
    "ci":  ("Deriv=CI",  "DerivPOS=NOUN"),
    "cı":  ("Deriv=CI",  "DerivPOS=NOUN"),
    "cu":  ("Deriv=CI",  "DerivPOS=NOUN"),
    "cü":  ("Deriv=CI",  "DerivPOS=NOUN"),
    "çı":  ("Deriv=CI",  "DerivPOS=NOUN"),
    "çi":  ("Deriv=CI",  "DerivPOS=NOUN"),
    "çu":  ("Deriv=CI",  "DerivPOS=NOUN"),
    "çü":  ("Deriv=CI",  "DerivPOS=NOUN"),
}
PLUR = {"lar": "Number=Plur", "ler": "Number=Plur"}
POSS_LONG = {
    "ımız": ("Poss=Yes", "Person[psor]=1", "Number[psor]=Plur"),
    "imiz": ("Poss=Yes", "Person[psor]=1", "Number[psor]=Plur"),
    "umuz": ("Poss=Yes", "Person[psor]=1", "Number[psor]=Plur"),
    "ümüz": ("Poss=Yes", "Person[psor]=1", "Number[psor]=Plur"),
    "ınız": ("Poss=Yes", "Person[psor]=2", "Number[psor]=Plur"),
    "iniz": ("Poss=Yes", "Person[psor]=2", "Number[psor]=Plur"),
    "unuz": ("Poss=Yes", "Person[psor]=2", "Number[psor]=Plur"),
    "ünüz": ("Poss=Yes", "Person[psor]=2", "Number[psor]=Plur"),
    "ları": ("Poss=Yes", "Person[psor]=3", "Number[psor]=Plur"),
    "leri": ("Poss=Yes", "Person[psor]=3", "Number[psor]=Plur"),
}
POSS_SHORT = {
    "ım": ("Poss=Yes", "Person[psor]=1", "Number[psor]=Sing"),
    "im": ("Poss=Yes", "Person[psor]=1", "Number[psor]=Sing"),
    "um": ("Poss=Yes", "Person[psor]=1", "Number[psor]=Sing"),
    "üm": ("Poss=Yes", "Person[psor]=1", "Number[psor]=Sing"),
    "ın": ("Poss=Yes", "Person[psor]=2", "Number[psor]=Sing"),
    "in": ("Poss=Yes", "Person[psor]=2", "Number[psor]=Sing"),
    "un": ("Poss=Yes", "Person[psor]=2", "Number[psor]=Sing"),
    "ün": ("Poss=Yes", "Person[psor]=2", "Number[psor]=Sing"),
    "sı": ("Poss=Yes", "Person[psor]=3", "Number[psor]=Sing"),
    "si": ("Poss=Yes", "Person[psor]=3", "Number[psor]=Sing"),
    "su": ("Poss=Yes", "Person[psor]=3", "Number[psor]=Sing"),
    "sü": ("Poss=Yes", "Person[psor]=3", "Number[psor]=Sing"),
    "m":  ("Poss=Yes", "Person[psor]=1", "Number[psor]=Sing"),
    "n":  ("Poss=Yes", "Person[psor]=2", "Number[psor]=Sing"),
}
CASE_ENDINGS = {
    "ndan": "Case=Abl", "nden": "Case=Abl",
    "dan": "Case=Abl", "den": "Case=Abl", "tan": "Case=Abl", "ten": "Case=Abl",
    "nda": "Case=Loc", "nde": "Case=Loc",
    "da":  "Case=Loc", "de":  "Case=Loc", "ta": "Case=Loc", "te": "Case=Loc",
    "nın": "Case=Gen", "nin": "Case=Gen", "nun": "Case=Gen", "nün": "Case=Gen",
    "ın":  "Case=Gen", "in":  "Case=Gen",  "un": "Case=Gen",  "ün":  "Case=Gen",
    "yla": "Case=Ins", "yle": "Case=Ins", "la": "Case=Ins", "le": "Case=Ins",
    "ca": "Case=Equ", "ce":  "Case=Equ",
    "yı": "Case=Acc", "yi": "Case=Acc", "yu": "Case=Acc", "yü": "Case=Acc",
    "ı":  "Case=Acc", "i":  "Case=Acc", "u":  "Case=Acc", "ü":  "Case=Acc",
    "ya": "Case=Dat", "ye": "Case=Dat", "a": "Case=Dat",  "e":  "Case=Dat",
}
BUFFER_N_ACC = {"nı": "Case=Acc", "ni": "Case=Acc", "nu": "Case=Acc", "nü": "Case=Acc"}
BUFFER_N_DAT = {"na": "Case=Dat", "ne": "Case=Dat"}

def is_other_token(tok: str) -> bool:
    if not tok: return True
    if URL_RE.match(tok) or MENTION_RE.match(tok) or HASHTAG_RE.match(tok): return True
    if NUMERIC_RE.match(tok): return True
    if EMOJI_RE.search(tok): return True
    if re.fullmatch(r"[\W_]+", tok): return True
    return False

def clean_token(token: str) -> str:
    return re.sub(r"[^\w’']+", "", token)

def tokenize(text: str):
    return re.findall(r"\w+['’]?\w*|\w+|['’]", text)

class Annotator:
    def __init__(self, freq_tr="frequent_tr_words.txt", freq_en="frequent_en_words.txt", ft_path="lid.176.ftz"):
        self.turkish_freq_top = set()
        self.turkish_freq_all = set()
        self.english_freq_words = set()
        with open(freq_tr, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts: continue
                w = parts[0].lower()
                self.turkish_freq_all.add(w)
                if i < 1000:
                    self.turkish_freq_top.add(w)
        with open(freq_en, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                self.english_freq_words.add(parts[0].lower())
        self.ft_model = fasttext.load_model(ft_path)
        self.ner = None

    @lru_cache(maxsize=200000)
    def _ft_predict(self, token_l: str):
        labels, probs = self.ft_model.predict(token_l, k=1)
        lang = labels[0].replace("__label__", "").upper()
        return lang, float(probs[0])

    def _choose_label(self, token_l: str, cfg):
        if token_l in self.turkish_freq_top:
            return "TR"
        if token_l in self.english_freq_words:
            return "EN"
        if token_l in self.turkish_freq_all:
            return "TR"
        lang, prob = self._ft_predict(token_l)
        if lang == "EN" and prob >= cfg["FT_EN_MIN"]: return "EN"
        if lang == "TR" and prob >= cfg["FT_TR_MIN"]: return "TR"
        return "UID"

    def _parse_tr_suffixes_full(self, suffix: str):
        s = suffix.lower()
        segments_rev, ud, deriv, amb = [], set(), set(), set()
        progressed = True
        while progressed and s:
            progressed = False
            for end, feat in sorted({**BUFFER_N_ACC, **BUFFER_N_DAT}.items(), key=lambda x: -len(x[0])):
                if s.endswith(end):
                    segments_rev.append(s[-len(end):])
                    ud.add(feat); s = s[:-len(end)]; progressed = True; break
            if progressed: continue
            for end, feat in sorted(CASE_ENDINGS.items(), key=lambda x: -len(x[0])):
                if s.endswith(end):
                    segments_rev.append(s[-len(end):])
                    ud.add(feat); s = s[:-len(end)]; progressed = True; break
        progressed = True
        while progressed and s:
            progressed = False
            for end, feats in sorted(POSS_LONG.items(), key=lambda x: -len(x[0])):
                if s.endswith(end):
                    segments_rev.append(s[-len(end):]); ud |= set(feats); s = s[:-len(end)]; progressed = True; break
            if progressed: continue
            for end, feats in sorted(POSS_SHORT.items(), key=lambda x: -len(x[0])):
                if s.endswith(end):
                    segments_rev.append(s[-len(end):]); ud |= set(feats); s = s[:-len(end)]; progressed = True; break
            if not progressed:
                for end in ["ı", "i", "u", "ü"]:
                    if s.endswith(end):
                        segments_rev.append(end); amb.add("Amb=P3sg_or_Acc"); s = s[:-1]; progressed = True; break
        for end, feat in sorted(PLUR.items(), key=lambda x: -len(x[0])):
            if s.endswith(end):
                segments_rev.append(end); ud.add(feat); s = s[:-len(end)]; break
        progressed = True
        while progressed and s:
            progressed = False
            for end, (dtag, dpos) in sorted(DERIV_SUFFIXES.items(), key=lambda x: -len(x[0])):
                if s.endswith(end):
                    segments_rev.append(end); deriv.add(dtag); deriv.add(dpos); s = s[:-len(end)]; progressed = True; break
        if s:
            segments_rev.append(s); deriv.add("Unparsed=Leftover")
        segments = list(reversed(segments_rev))
        return segments, ud, deriv, amb

    def _split_mixed_apostrophe(self, token: str):
        if "'" in token or "’" in token:
            parts = re.split(r"[’']", token)
            if len(parts) == 2 and parts[0] and parts[1]:
                base, suff = parts
                sfx = suff.lower()
                if sfx in EN_CONTRACTIONS or sfx.endswith("n't"):
                    return None, None
                return base, suff
        return None, None

    def _detect_mixed_no_apostrophe(self, token: str, cfg):
        tok = token
        tok_l = tok.lower()
        if tok_l in self.turkish_freq_all:
            return None, None
        all_suffixes = (set(CASE_ENDINGS.keys()) | set(PLUR.keys()) |
                        set(POSS_LONG.keys()) | set(POSS_SHORT.keys()) |
                        set(DERIV_SUFFIXES.keys()) | set(BUFFER_N_ACC.keys()) | set(BUFFER_N_DAT.keys()))
        for suf in sorted(all_suffixes, key=len, reverse=True):
            if len(suf) < 2: continue
            if tok_l.endswith(suf):
                base = tok[:-len(suf)]
                base_clean = clean_token(base).lower()
                if len(base_clean) < 2: continue
                base_is_en = False
                if base_clean in self.english_freq_words:
                    base_is_en = True
                else:
                    blang, bprob = self._ft_predict(base_clean)
                    if blang == "EN" and bprob >= cfg["FT_EN_MIN"]:
                        base_is_en = True
                if not base_is_en: continue
                last_char = base_clean[-1] if base_clean else ""
                last_is_vowel = last_char in "aeıioöuü"
                if suf[0] in ("y", "n") and not last_is_vowel:
                    continue
                segments, ud_feats, deriv, amb = self._parse_tr_suffixes_full(suf)
                has_ud = bool(ud_feats or deriv or amb)
                if not has_ud: continue
                if cfg["MIXED_STRICT"] and base_clean in self.turkish_freq_all:
                    continue
                return base, token[-len(suf):]
        return None, None

    def _build_ne_map(self, doc, line_tokens):
        ne_map = {}
        if not getattr(doc, "ents", None): return ne_map
        ne_pieces = set()
        for ent in doc.ents:
            for piece in re.findall(r"\w+['’]?\w*|\w+", ent.text):
                if piece: ne_pieces.add(piece)
        for tok in line_tokens:
            if tok in ne_pieces:
                ne_map[tok] = "NE"
        return ne_map

    def _ensure_ner(self, enabled=True):
        if enabled and self.ner is None:
            self.ner = stanza.Pipeline("tr", processors="tokenize,ner", use_gpu=False)

    def _decide_matrix_embed(self, labels, cfg):
        """
        Basit ve deterministik kural:
          - Matrix: TR/EN oyla (MIXED'i ağırlıklandır)
          - Embed:
              Matrix TR ise: cümlede EN veya MIXED varsa EN, yoksa "-"
              Matrix EN ise: cümlede TR veya MIXED varsa TR, yoksa "-"
        """
        # Matrix
        score_tr = sum(1 for lb in labels if lb == "TR")
        score_en = sum(1 for lb in labels if lb == "EN")
        mixed_cnt = sum(1 for lb in labels if lb == "MIXED")
        score_tr += cfg["MIXED_TR_WEIGHT"] * mixed_cnt
        score_en += cfg["MIXED_EN_WEIGHT"] * mixed_cnt

        # Eşitlikte TR'yi tercih ediyoruz (önceki davranışla uyumlu)
        matrix = "TR" if score_tr >= score_en else "EN"

        # Embed kararı
        if matrix == "TR":
            # EN veya MIXED var mı?
            embed = "EN" if any(lb in ("EN", "MIXED") for lb in labels) else "-"
        else:
            # TR veya MIXED var mı?
            embed = "TR" if any(lb in ("TR", "MIXED") for lb in labels) else "-"

        return matrix, embed

    def annotate(self, text, user_cfg=None):
        cfg = DEFAULTS.copy()
        if user_cfg: cfg.update(user_cfg)
        self._ensure_ner(cfg["NER_ENABLED"])

        lines = text.splitlines()
        out_lines = []
        sent_idx = 0

        for raw in lines:
            line = raw.rstrip("\n")
            if not line.strip():
                out_lines.append("")
                continue

            # Cümle numarası
            sent_idx += 1
            if cfg.get("FEATURE_SENTENCE_ID", True):
                out_lines.append(f"SentenceID\t{sent_idx}")

            ner_doc = None
            tokens = tokenize(line)
            if cfg["NER_ENABLED"]:
                ner_doc = self.ner(line)
                ne_map = self._build_ne_map(ner_doc, tokens)
            else:
                ne_map = {}

            labels_in_sent = []
            sent_rows = []

            for tok in tokens:
                tok_clean = clean_token(tok)
                if is_other_token(tok):
                    if cfg["FEATURE_LANGUAGE_PER_ITEM"]:
                        sent_rows.append(f"{tok}\tOTHER")
                    continue

                if tok in ne_map:
                    if cfg["FEATURE_LANGUAGE_PER_ITEM"]:
                        sent_rows.append(f"{tok}\tNE")
                        labels_in_sent.append("NE")
                    continue

                if not cfg["FEATURE_LANGUAGE_PER_ITEM"]:
                    # sadece matrix/embed için işlem
                    tok_l = tok_clean.lower()
                    label = self._choose_label(tok_l, cfg)
                    base, suf = self._split_mixed_apostrophe(tok)
                    if base and suf:
                        base_l = clean_token(base).lower()
                        base_lb = self._choose_label(base_l, cfg)
                        if base_lb == "EN":
                            segments, ud_feats, deriv, amb = self._parse_tr_suffixes_full(suf)
                            if (ud_feats or deriv or amb):
                                labels_in_sent.append("MIXED")
                                continue
                    if label != "TR":
                        base2, suf2 = self._detect_mixed_no_apostrophe(tok, cfg)
                        if base2 and suf2:
                            labels_in_sent.append("MIXED")
                            continue
                    if label in ("TR", "EN"): labels_in_sent.append(label)
                    continue

                tok_l = tok_clean.lower()
                label = self._choose_label(tok_l, cfg)

                base, suf = self._split_mixed_apostrophe(tok)
                if base and suf:
                    base_l = clean_token(base).lower()
                    base_label = self._choose_label(base_l, cfg)
                    if base_label == "EN":
                        segments, ud_feats, deriv, amb = self._parse_tr_suffixes_full(suf)
                        if (ud_feats or deriv or amb):
                            sent_rows.append(f"{tok}\tMIXED")
                            labels_in_sent.append("MIXED")
                            continue
                    if base_label == "TR":
                        sent_rows.append(f"{tok}\tTR")
                        labels_in_sent.append("TR")
                        continue

                if label != "TR":
                    base2, suf2 = self._detect_mixed_no_apostrophe(tok, cfg)
                    if base2 and suf2:
                        sent_rows.append(f"{tok}\tMIXED")
                        labels_in_sent.append("MIXED")
                        continue

                sent_rows.append(f"{tok}\t{label}")
                if label in ("TR", "EN", "MIXED"):
                    labels_in_sent.append(label)

            out_lines.extend(sent_rows)

            # Matrix/Embed yazımı
            matrix, embed = self._decide_matrix_embed(labels_in_sent, cfg)

            if cfg["FEATURE_MATRIX_LANGUAGE"]:
                out_lines.append(f"MatrixLang\t{matrix}")
                if cfg["FEATURE_EMBEDDED_LANGUAGE"]:
                    # Her zaman yaz
                    out_lines.append(f"EmbedLang\t{embed}")
            else:
                # Matrix istenmiyorsa bile Embed seçildiyse yine hesapla ve yaz
                if cfg["FEATURE_EMBEDDED_LANGUAGE"]:
                    out_lines.append(f"EmbedLang\t{embed}")

            out_lines.append("")

        return "\n".join(out_lines)
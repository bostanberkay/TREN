"""Check that the installed fasttext works with TREN's language-ID model.

Importing fasttext is not enough: this loads src/tren/resources/lid.176.ftz and runs
predict() the way cs_pipeline.Annotator does (k=1), which is also where an
incompatible numpy fails. The top label must equal, and the probability be
within 1e-4 of, the reference values below, recorded with fasttext 0.9.3 and
numpy 1.26.4 on macOS arm64. Exit code 0 means every prediction matched.

    python packaging/check_fasttext.py
"""
import importlib.metadata
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "src", "tren", "resources", "lid.176.ftz")
EXPECTED_VERSION = "0.9.3"
TOLERANCE = 1e-4
REFERENCE = [
    ("merhaba", "__label__tr", 0.7364962100982666),
    ("hello", "__label__en", 0.24247202277183533),
    ("kitap", "__label__tr", 0.36670994758605957),
    ("meeting", "__label__en", 0.7159436941146851),
    ("deadline", "__label__en", 0.688125729560852),
    ("bonjour", "__label__fr", 0.9015306830406189),
    ("güzel", "__label__tr", 0.9819150567054749),
    ("computer", "__label__en", 0.9072192311286926),
    ("teşekkürler", "__label__tr", 0.9231404066085815),
    ("ankara'dan", "__label__tr", 0.992559015750885),
]


def main():
    # A redirected stdout on Windows uses the ANSI code page, which lacks e.g. "ş".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    import fasttext
    import numpy

    version = importlib.metadata.version("fasttext")
    print(f"fasttext {version} from {fasttext.__file__}; numpy {numpy.__version__}")
    failures = []
    if version != EXPECTED_VERSION:
        failures.append(f"fasttext version {version}, expected {EXPECTED_VERSION}")

    model = fasttext.load_model(MODEL)
    for token, label, prob in REFERENCE:
        labels, probs = model.predict(token, k=1)
        got_label, got_prob = labels[0], float(probs[0])
        ok = got_label == label and abs(got_prob - prob) <= TOLERANCE
        print(f"{'ok  ' if ok else 'FAIL'} {token!r}: {got_label} {got_prob:.6f} (expected {label} {prob:.6f})")
        if not ok:
            failures.append(token)

    if failures:
        print(f"fastText check FAILED: {failures}")
        return 1
    print(f"fastText check passed: {len(REFERENCE)} predictions match the reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

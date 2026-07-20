# examples/quickstart.py
"""Minimal, deterministic quickstart example for cs_pipeline.Annotator.

Runs the real Annotator.annotate() pipeline on a short Turkish-English
code-switching sentence, without loading any real Stanza or fastText
model. Instantiation bypasses Annotator.__init__ (which would otherwise
read the real frequency-list files and load a fastText binary model) via
Annotator.__new__, using small synthetic Turkish/English lexicon sets
instead. Only the fastText prediction call (_ft_predict) is replaced --
the narrowest possible boundary -- so tokenization, suffix parsing, MIXED
detection, Matrix/Embedded Language voting, and final output construction
all run as real, unmodified code.

Run from the repository root:
    python examples/quickstart.py
"""

import os
import sys
from unittest.mock import patch

# Running `python examples/quickstart.py` puts the examples/ directory,
# not the repository root, on sys.path by default. Add the repo root so
# `import cs_pipeline` resolves regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs_pipeline import Annotator, DEFAULTS

INPUT_TEXT = "kitap amazing boss'um"

EXPECTED_OUTPUT = (
    "SentenceID\t1\n"
    "kitap\tTR\n"
    "amazing\tEN\n"
    "boss'um\tMIXED\n"
    "MatrixLang\tTR\n"
    "EmbedLang\tEN\n"
)


def main():
    annotator = Annotator.__new__(Annotator)
    annotator.turkish_freq_top = {"kitap"}
    annotator.turkish_freq_all = set()
    annotator.english_freq_words = {"amazing", "boss"}

    cfg = dict(DEFAULTS, NER_ENABLED=False)

    with patch.object(annotator, "_ft_predict", return_value=("UID", 0.0)):
        output = annotator.annotate(INPUT_TEXT, cfg)

    print("Input:")
    print(INPUT_TEXT)
    print()
    print("Annotation output:")
    print(output)

    if output != EXPECTED_OUTPUT:
        print("FAILED: annotation output did not match the expected output.", file=sys.stderr)
        print("--- expected ---", file=sys.stderr)
        print(repr(EXPECTED_OUTPUT), file=sys.stderr)
        print("--- actual ---", file=sys.stderr)
        print(repr(output), file=sys.stderr)
        sys.exit(1)

    print("OK: quickstart annotation matches the expected output.")


if __name__ == "__main__":
    main()

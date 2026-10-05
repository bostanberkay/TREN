"""Compatibility launcher for `python cs_annotator_app.py` in a source checkout.

TREN's code is in src/tren/; the usual command is `python -m tren` after
`pip install .` (see README). This file only puts src/ on the import path and
calls that same entry point.
"""
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from tren.cs_annotator_app import main

    main()

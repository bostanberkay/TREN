"""Tk start-up under pytest's two capture modes; not part of the normal suite
(the file name does not match test_*.py), run only by naming it explicitly:

    python -m pytest tests/tk_capture_stress.py --capture=fd
    python -m pytest tests/tk_capture_stress.py --capture=sys

Each test creates and destroys one Tk root, with pytest's capture switched
around it as for any test. On Windows, fd capture is expected to make some of
these fail with 'invalid command name "tcl_findLibrary"' (see pytest.ini); sys
capture must not. TREN_TK_STRESS_ROOTS sets the number of roots (default 1000).
Used by .github/workflows/windows-tk-check.yml.
"""
import os
import tkinter

import pytest

ROOTS = int(os.environ.get("TREN_TK_STRESS_ROOTS", "1000"))


@pytest.mark.parametrize("n", range(ROOTS))
def test_tk_root_starts(n):
    root = tkinter.Tk()
    root.destroy()
    print(n)

"""Tk start-up probe shared by the GUI test modules.

Each GUI test module calls gui_skip(__name__) when it is imported, i.e. during
pytest's collection. That creates and destroys one Tk root and keeps the real
exception and traceback if this fails, instead of swallowing them:

* by default a failure skips the module, with the exception as the reason;
* with TREN_REQUIRE_TK=1 (set in CI) nothing is skipped: the module's tests
  run against real Tk, and tests/test_tk_startup.py fails with every recorded
  traceback, so a Tk start-up failure can never pass silently.

conftest.py prints the probe records in pytest's terminal summary when one failed.
"""
import os
import sys
import textwrap
import traceback

REQUIRE_ENV = "TREN_REQUIRE_TK"
RECORDS = []


def tk_required():
    return os.environ.get(REQUIRE_ENV) == "1"


def probe(module):
    """Create and destroy one Tk root and record the outcome; returns the exception
    line, or None if it worked."""
    record = {"module": module, "error": None, "traceback": None}
    try:
        import tkinter
        root = tkinter.Tk()
        root.destroy()
    except Exception as e:
        record["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
        record["traceback"] = traceback.format_exc()
    RECORDS.append(record)
    return record["error"]


def gui_skip(module):
    """(skip, reason) for a GUI test module's skipif mark; see the module docstring."""
    error = probe(module)
    if error is None or tk_required():
        return False, ""
    return True, (f"Tk could not start while collecting {module}: {error} (traceback under "
                  f"'Tk start-up probes' in the pytest summary; set {REQUIRE_ENV}=1 to fail instead)")


def failed_records():
    return [r for r in RECORDS if r["error"] is not None]


def summary_lines():
    lines = []
    try:
        import tkinter
        lines.append(f"Tcl/Tk {tkinter.Tcl().eval('info patchlevel')}, Python {sys.version.split()[0]}, {sys.platform}")
    except Exception as e:
        lines.append(f"could not query Tcl/Tk: {e!r}")
    for n, r in enumerate(RECORDS, 1):
        lines.append(f"#{n} {'ok' if r['error'] is None else 'FAILED'}: {r['module']}")
        if r["traceback"]:
            lines.append(textwrap.indent(r["traceback"].rstrip(), "    "))
    return lines

"""Tk start-up probe shared by the GUI test modules, plus a diagnostic CLI.

Each GUI test module calls gui_skip(__name__) when it is imported, i.e. during
pytest's collection. That creates and destroys one Tk root, as each module's
own _tk_available() used to, but keeps the real exception and traceback
instead of swallowing them:

* by default a failure skips the module, with the traceback as the reason;
* with TREN_REQUIRE_TK=1 (set in CI) nothing is skipped: the module's tests
  run against real Tk, and tests/test_tk_startup.py fails with every recorded
  traceback, so a Tk start-up failure can never pass silently.

conftest.py prints every probe record in pytest's terminal summary when a probe
failed or TREN_TK_DIAGNOSTICS=1. A failed probe is retried once at once, for the
record only; the retry never changes the outcome. On Windows each record also
lists this process's top-level window classes before and after the probe.

    python tests/tk_probe.py [--repeat N]   # N create/destroy cycles outside pytest
"""
import os
import sys
import textwrap
import threading
import time
import traceback

REQUIRE_ENV = "TREN_REQUIRE_TK"
DIAGNOSTICS_ENV = "TREN_TK_DIAGNOSTICS"
RECORDS = []
_START = time.perf_counter()


def tk_required():
    return os.environ.get(REQUIRE_ENV) == "1"


def diagnostics_requested():
    return os.environ.get(DIAGNOSTICS_ENV) == "1"


def _windows_toplevel_classes():
    """{window class: count} of this process's top-level windows; None off Windows.
    Diagnostic only, so it never raises (that would break collection)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32")  # private instance: argtypes set here stay local
        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows.argtypes = [enum_proc, wintypes.LPARAM]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        pid = os.getpid()
        counts = {}

        def visit(hwnd, _lparam):
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid:
                name = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(hwnd, name, 256)
                key = name.value + ("" if user32.IsWindowVisible(hwnd) else " (hidden)")
                counts[key] = counts.get(key, 0) + 1
            return True

        user32.EnumWindows(enum_proc(visit), 0)
        return counts
    except Exception as e:
        return {"<window enumeration failed>": repr(e)}


def _create_and_destroy_root():
    """None if a Tk root could be created and destroyed, else (exception line, traceback)."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.destroy()
        return None
    except Exception as e:
        return "".join(traceback.format_exception_only(type(e), e)).strip(), traceback.format_exc()


def probe(module):
    """Create and destroy one Tk root and record the outcome; returns the exception
    line, or None if it worked."""
    tkinter = sys.modules.get("tkinter")
    record = {
        "n": len(RECORDS) + 1,
        "module": module,
        "t": round(time.perf_counter() - _START, 3),
        "threads": [t.name for t in threading.enumerate()],
        "default_root_alive": getattr(tkinter, "_default_root", None) is not None,
        "windows_before": _windows_toplevel_classes(),
    }
    record["error"] = _create_and_destroy_root()
    if record["error"] is not None:
        record["retry_error"] = _create_and_destroy_root()
    record["windows_after"] = _windows_toplevel_classes()
    RECORDS.append(record)
    return record["error"] and record["error"][0]


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
        patchlevel = tkinter.Tcl().eval("info patchlevel")
        lines.append(f"Tcl/Tk {patchlevel}, Python {sys.version.split()[0]}, {sys.platform}, cwd {os.getcwd()}")
    except Exception as e:
        lines.append(f"could not query Tcl/Tk: {e!r}")
    for r in RECORDS:
        status = "ok" if r["error"] is None else "FAILED"
        lines.append(f"#{r['n']} {status}: {r['module']} at {r['t']} s; threads={r['threads']}; "
                     f"default root alive={r['default_root_alive']}")
        if r["windows_before"] is not None:
            lines.append(f"    top-level windows before={r['windows_before']} after={r['windows_after']}")
        if r["error"] is not None:
            lines.append(textwrap.indent(r["error"][1].rstrip(), "    "))
            retry = r["retry_error"]
            lines.append("    immediate retry: " + ("succeeded" if retry is None else "failed as well:"))
            if retry is not None:
                lines.append(textwrap.indent(retry[1].rstrip(), "    "))
    return lines


def main(argv):
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 10
    for i in range(repeat):
        probe(f"standalone cycle {i + 1}")
    print("\n".join(summary_lines()))
    failed = failed_records()
    print(f"{repeat - len(failed)} of {repeat} Tk create/destroy cycles succeeded.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Reports Tk start-up failures recorded while the GUI test modules were
collected (see tk_probe.py): with TREN_REQUIRE_TK=1 (CI) this test fails with
every recorded traceback; otherwise it is skipped with them, like the GUI
modules themselves.
"""
import pytest

import tk_probe


def test_tk_started_for_every_gui_module():
    if not tk_probe.RECORDS:  # run on its own: no GUI module was collected
        tk_probe.probe(__name__)
    failed = tk_probe.failed_records()
    if not failed:
        return
    modules = ", ".join(r["module"] for r in failed)
    message = f"Tk failed to start while collecting: {modules}\n" + "\n".join(tk_probe.summary_lines())
    if tk_probe.tk_required():
        pytest.fail(message, pytrace=False)
    pytest.skip(message)

import sys
from unittest.mock import MagicMock

# cs_pipeline.py does `import stanza` unconditionally at module level, even
# though NER is a togglable feature. Stub it here, before any test module
# imports cs_pipeline, so the test suite doesn't require the real (heavy)
# stanza package to be installed for tests that never touch NER.
sys.modules["stanza"] = MagicMock()

import tk_probe  # noqa: E402


def pytest_terminal_summary(terminalreporter):
    """List every Tk start-up probe when one failed or TREN_TK_DIAGNOSTICS=1."""
    if tk_probe.RECORDS and (tk_probe.failed_records() or tk_probe.diagnostics_requested()):
        terminalreporter.section("Tk start-up probes (tests/tk_probe.py)")
        for line in tk_probe.summary_lines():
            terminalreporter.write_line(line)

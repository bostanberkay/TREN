import sys
from unittest.mock import MagicMock

# cs_pipeline.py does `import stanza` unconditionally at module level, even
# though NER is a togglable feature. Stub it here, before any test module
# imports cs_pipeline, so the test suite doesn't require the real (heavy)
# stanza package to be installed for tests that never touch NER.
sys.modules["stanza"] = MagicMock()

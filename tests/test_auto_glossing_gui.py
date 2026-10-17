"""GUI tests for the Auto-Glossing Tool: the gloss suggestions for MIXED tokens and the
tool's Auto-Gloss, navigate, and close flow, which commits glosses to the model and the
grid. Suggestions are editable proposals; only unambiguous suffixes are asserted here,
so these tests check behavior, not the linguistic accuracy of the suggestions.
Real Tk; skipped without a display (CI runs it under xvfb).
"""
import pytest


from tren import cs_annotator_app as caa  # noqa: E402
import tk_probe  # noqa: E402


# Probed during collection; the real traceback is kept (see tk_probe.py).
_SKIP_GUI, _SKIP_REASON = tk_probe.gui_skip(__name__)
pytestmark = pytest.mark.skipif(_SKIP_GUI, reason=_SKIP_REASON)


@pytest.fixture(autouse=True)
def _isolated_app_dir(monkeypatch, tmp_path):
    app_dir = tmp_path / "cs_annotator_home"
    monkeypatch.setattr(caa, "APP_DIR", str(app_dir))
    monkeypatch.setattr(caa, "LAST_PROJECT_PTR", str(app_dir / "last_project.json"))


TWO_MIXED = (
    "SentenceID\t1\nkitap\tTR\nboss'um\tMIXED\nMatrixLang\tTR\nEmbedLang\tEN\n\n"
    "SentenceID\t2\nmeeting'e\tMIXED\nbugün\tTR\nMatrixLang\tTR\nEmbedLang\tEN\n"
)
NO_MIXED = "SentenceID\t1\nkitap\tTR\nMatrixLang\tTR\nEmbedLang\t-\n"
MIXED_WITHOUT_SUFFIX = "SentenceID\t1\nkitap\tMIXED\nMatrixLang\tTR\nEmbedLang\t-\n"


class _Dialogs:
    """Records every modal messagebox call; each returns a preset answer, so
    no test can block on a real dialog."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.answers = {"askokcancel": True, "askyesnocancel": None, "askyesno": True}
        for name in ("showwarning", "showerror", "showinfo",
                     "askokcancel", "askyesnocancel", "askyesno"):
            monkeypatch.setattr(caa.messagebox, name, self._make(name))

    def _make(self, name):
        def _dialog(*a, **k):
            self.calls.append(name)
            return self.answers.get(name)
        return _dialog


def make_app(monkeypatch, pipeline_output):
    dialogs = _Dialogs(monkeypatch)
    app = caa.App()
    app.update()
    monkeypatch.setattr(app, "_run_annotation_pipeline", lambda text: pipeline_output)
    monkeypatch.setattr(app, "_attach_confidence", lambda blocks: None)
    return app, dialogs


def run_once(app, text):
    app._set_txt_input_text(text)
    app.run_pipeline()
    app.update()


def model_row(app, token):
    return next(r for b in app.blocks for r in b if r.get("token") == token)


def grid_gloss(app, token):
    vis = next(v for v, (b, r) in app._row_index_map.items()
               if b is not None and app.blocks[b][r].get("token") == token)
    return app.sheet.get_cell_data(vis, 3)


def open_tool(app):
    app.open_auto_glossing_tool()
    app.update()


# --- suggestions ---------------------------------------------------------

def test_suggestions_for_unambiguous_suffixes(monkeypatch):
    app, _ = make_app(monkeypatch, NO_MIXED)
    try:
        expected = {
            "boss'um": "boss-POSS.1SG",  # the README's example
            "meeting'e": "meeting-DAT",
            "office'te": "office-LOC",
            "office'ten": "office-ABL",
            "meeting'ler": "meeting-PL",
            "meeting'lerde": "meeting-PL-LOC",
        }
        for token, gloss in expected.items():
            assert app._auto_gloss_mixed_token(token) == gloss, token
    finally:
        app.destroy()


def test_ambiguous_suffix_lists_every_gloss_and_suggests_the_first(monkeypatch):
    app, _ = make_app(monkeypatch, NO_MIXED)
    try:
        candidates = app._auto_gloss_candidates("meeting'le")
        assert sorted(candidates) == ["meeting-COM", "meeting-INS"]
        assert app._auto_gloss_mixed_token("meeting'le") == candidates[0]
    finally:
        app.destroy()


def test_no_suggestion_without_a_recognized_suffix(monkeypatch):
    app, _ = make_app(monkeypatch, NO_MIXED)
    try:
        for token in ("kitap", "amazing", "", "  ", None):
            assert app._auto_gloss_candidates(token) == [], repr(token)
            assert app._auto_gloss_mixed_token(token) == "", repr(token)
    finally:
        app.destroy()


# --- the tool ------------------------------------------------------------

def test_tool_lists_mixed_tokens_and_commits_glosses_to_model_and_grid(monkeypatch):
    app, dialogs = make_app(monkeypatch, TWO_MIXED)
    try:
        run_once(app, "kitap boss'um\n\nmeeting'e bugün")
        open_tool(app)
        tokens = [app.blocks[i["bidx"]][i["ridx"]]["token"] for i in app._ag_items]
        assert tokens == ["boss'um", "meeting'e"]
        assert app._ag_item_var.get() == "boss'um"
        assert app._ag_label_var.get() == "MIXED"
        assert app._ag_gloss_var.get() == ""

        app._ag_auto_gloss_current()
        assert app._ag_gloss_var.get() == "boss-POSS.1SG"
        assert model_row(app, "boss'um")["gloss"] == ""  # nothing is committed before navigating away

        app._ag_next()
        assert app._ag_item_var.get() == "meeting'e"
        assert model_row(app, "boss'um")["gloss"] == "boss-POSS.1SG"
        assert grid_gloss(app, "boss'um") == "boss-POSS.1SG"

        app._ag_auto_gloss_current()
        app._ag_prev()
        assert model_row(app, "meeting'e")["gloss"] == "meeting-DAT"
        assert grid_gloss(app, "meeting'e") == "meeting-DAT"
        assert app._ag_item_var.get() == "boss'um"
        assert app._ag_gloss_var.get() == "boss-POSS.1SG"  # the committed gloss is shown again
        assert dialogs.calls == []
    finally:
        app.destroy()


def test_closing_the_tool_commits_the_current_gloss(monkeypatch):
    app, _ = make_app(monkeypatch, TWO_MIXED)
    try:
        run_once(app, "kitap boss'um\n\nmeeting'e bugün")
        open_tool(app)
        app._ag_auto_gloss_current()
        win = app._ag_win
        app.tk.call(win.protocol("WM_DELETE_WINDOW"))
        app.update()
        assert model_row(app, "boss'um")["gloss"] == "boss-POSS.1SG"
        assert grid_gloss(app, "boss'um") == "boss-POSS.1SG"
        assert app._ag_win is None
    finally:
        app.destroy()


def test_auto_gloss_keeps_a_typed_gloss_when_there_is_no_suggestion(monkeypatch):
    app, _ = make_app(monkeypatch, MIXED_WITHOUT_SUFFIX)
    try:
        run_once(app, "kitap")
        open_tool(app)
        app._ag_gloss_var.set("typed-by-hand")
        app._ag_auto_gloss_current()
        assert app._ag_gloss_var.get() == "typed-by-hand"
    finally:
        app.destroy()


def test_auto_gloss_on_a_row_relabeled_away_from_mixed_only_rings_the_bell(monkeypatch):
    app, _ = make_app(monkeypatch, TWO_MIXED)
    try:
        run_once(app, "kitap boss'um\n\nmeeting'e bugün")
        open_tool(app)
        model_row(app, "boss'um")["label"] = "TR"
        bells = []
        monkeypatch.setattr(app, "bell", lambda *a, **k: bells.append(1))
        app._ag_auto_gloss_current()
        assert bells == [1]
        assert app._ag_gloss_var.get() == ""
    finally:
        app.destroy()


def test_tool_without_mixed_tokens_says_so_and_changes_nothing(monkeypatch):
    app, dialogs = make_app(monkeypatch, NO_MIXED)
    try:
        run_once(app, "kitap")
        open_tool(app)
        assert dialogs.calls == ["showinfo"]
        assert app._ag_items == []
        app._ag_auto_gloss_current()
        assert model_row(app, "kitap")["gloss"] == ""
    finally:
        app.destroy()

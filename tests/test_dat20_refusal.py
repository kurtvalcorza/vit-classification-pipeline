"""DAT19/DAT20: a refused BYOD input renders exactly one `Refused: <file>: <reason>` line, no traceback.

Executes the committed Section 4 cell with the real loader and splitter. Covers the scripted hosted refusal
input (`BYOD_PATH='/etc/hostname'`: a file that is neither a folder nor a zip), a folder without labels.csv,
a missing path, a cancelled upload, the in-cell invalidation, and that a genuine error keeps its own type.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from test_notebook_review_fixes import NOTEBOOK, _section_4, _section_4_namespace

REFUSAL_CELL = "vit_classification-18"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _refused(notebook, path: str, ns: dict | None = None) -> str:
    ns = ns if ns is not None else _section_4_namespace([])
    with pytest.raises(Exception) as caught:
        exec(_section_4(notebook, path), ns)
    assert type(caught.value).__name__ == "RunStopped" and caught.value.dimer_stop is True
    message = str(caught.value)  # the router shows str(exc) alone for a dimer_stop exception
    assert "\n" not in message and message.startswith("Refused: ") and "Traceback" not in message
    assert ns["RUN_VALID"] is False
    return message


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    return tmp_path


def test_scripted_hosted_input_a_file_that_is_not_a_zip(notebook, workdir):
    (workdir / "hostname").write_text("colab-vm\n", encoding="utf-8")
    line = _refused(notebook, str(workdir / "hostname"))
    assert line.startswith("Refused: hostname: BYOD datasets must be a directory or a .zip"), line


def test_a_folder_without_labels_and_a_missing_path(notebook, workdir):
    (workdir / "etc").mkdir()
    assert _refused(notebook, str(workdir / "etc")).startswith(
        "Refused: etc: BYOD folder must contain labels.csv"
    )
    assert _refused(notebook, "nowhere.zip").startswith("Refused: nowhere.zip: BYOD_PATH does not exist")


def test_a_cancelled_upload(notebook, workdir, monkeypatch):
    google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
    files.upload = lambda: {}
    colab.files, google.colab = files, colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)
    line = _refused(notebook, "")
    assert line.startswith("Refused: (no file): upload exactly one .zip file (received 0"), line


def test_refusal_moves_earlier_outputs_aside(notebook, workdir):
    (workdir / "outputs").mkdir()
    (workdir / "outputs" / "vit_classification_evaluation_report.json").write_text("{}", encoding="utf-8")
    line = _refused(notebook, "nowhere.zip")
    assert "earlier files were moved out of outputs/" in line
    assert not list((workdir / "outputs").iterdir()) and list(workdir.glob("outputs_before_byod_*"))


def test_genuine_error_keeps_its_type(notebook, workdir):
    (workdir / "mine").mkdir()
    ns = _section_4_namespace([])

    def broken(path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug"):
        exec(_section_4(notebook, str(workdir / "mine")), ns)


def test_refusal_cell_id(notebook):
    ids = [c["id"] for c in notebook["cells"] if "class RunStopped" in "".join(c["source"])]
    assert ids == [REFUSAL_CELL]

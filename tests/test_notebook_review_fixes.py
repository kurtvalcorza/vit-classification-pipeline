"""Regression tests for the 2026-10-05 notebook review findings (VIT-M1..M3, VIT-m1..m4).

Every test needs only CI's dependencies and no model: the notebook's own cell sources are executed with stand-ins
where a model would be needed, and restore_base() is exercised on a stand-in model, not the checkpoint. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vit_classification_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "vit_classification_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "vit_classification_pipeline" / "pipeline.py"
STEM = "vit_classification"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- VIT-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_vit_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_vit_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)


def test_vit_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker — with every variable later cells created — is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- VIT-M2: every adaptation starts from the pinned base -------------------------------------------------------


def test_vit_m2_adapt_and_load_artifact_restore_the_base_first():
    """Torch-backed, so the order inside adapt is checked statically: pre-call state kept, base restored, then the probe."""
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def save_artifact(")]
    order = [adapt.index(m) for m in ("previous_state = {", "restored = self.restore_base()", "self._remember_base(names)", "x_train = torch.tensor(self.features(train_checked)", "for epoch in range(1, epochs + 1):")]
    assert order == sorted(order)
    failure = adapt[adapt.index("except BaseException:") :]
    assert failure.index("restore.update(initial_blocks)") < failure.index("restore.update(previous_state)") < failure.index("raise")
    assert '"started_from": "pinned base"' in adapt
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("self._remember_base(sorted(block_tensors))") < load.index("model.load_state_dict(merged")


class _Tensor:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)

    def detach(self):
        return self

    def clone(self):
        return _Tensor(self.value.copy())


class _Model:
    def __init__(self):
        self.state = {"blocks.11.w": _Tensor([1.0, 2.0]), "blocks.10.w": _Tensor([3.0]), "patch_embed.w": _Tensor([4.0])}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, values, strict=True):
        assert strict is True and set(values) == set(self.state)
        self.state = {name: _Tensor(tensor.value.copy()) for name, tensor in values.items()}

    def eval(self):
        return self


def test_vit_m2_restore_base_undoes_every_earlier_change_stand_in():
    """restore_base() and _remember_base() on a stand-in model with the state_dict interface (numpy, not torch)."""
    from vit_classification_pipeline import ViTClassificationPipeline

    model = _Model()
    pipe = ViTClassificationPipeline(_runner=lambda batch: None, _transform=lambda image: None, _model=model)
    assert pipe.restore_base() == []
    pipe._remember_base(["blocks.11.w", "blocks.10.w"])
    model.state["blocks.11.w"] = _Tensor([9.0, 9.0])
    model.state["blocks.10.w"] = _Tensor([9.0])
    pipe._remember_base(["blocks.11.w"])  # a second run keeps the first (base) value
    pipe._head, pipe.classes, pipe.adapter = object(), ["a", "b"], {"policy": "x"}
    assert pipe.restore_base() == ["blocks.10.w", "blocks.11.w"]
    assert model.state["blocks.11.w"].value.tolist() == [1.0, 2.0] and model.state["blocks.10.w"].value.tolist() == [3.0]
    assert model.state["patch_embed.w"].value.tolist() == [4.0]
    assert (pipe._head, pipe.classes, pipe.adapter) == (None, None, None)


def test_vit_m2_byod_rerun_restores_the_base_and_the_experiment_has_its_own_pipeline(notebook):
    section_4 = _cell(notebook, "USE_BYOD = False")
    assert section_4.index("restored_blocks = pipe.restore_base()") < section_4.index("if USE_BYOD:")
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "experiment_pipe = ViTClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)" in experiment
    assert f"Path('outputs/{STEM}_experiment')" in experiment
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in experiment
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", experiment)
    assert "**Predict → Change one thing → Run → Observe → Explain**" in _markdown(notebook)
    assert "they do not affect the default path" not in _markdown(notebook)


# --- VIT-M3: guided layer and infrastructure labelling ----------------------------------------------------------


def test_vit_m3_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "## 10. Change one thing", "**Learner:**"):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6


def test_vit_m3_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    assert len([c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]) == 3
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled


def test_vit_m3_no_template_placeholders_leak(notebook):
    learner = "\n".join(c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in learner, leftover
    assert "}}" not in _markdown(notebook)


# --- VIT-m1: BYOD contract --------------------------------------------------------------------------------------


def _jpeg(i: int, corrupt: bool = False) -> bytes:
    from PIL import Image

    if corrupt:
        return b"not an image"
    image = Image.fromarray(np.random.default_rng(i).integers(0, 255, (40, 48, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


def _zip(path: Path, per_label: int, labels=("cat", "dog"), *, drop: str | None = None, corrupt: str | None = None, extra: dict[str, bytes] | None = None, bom: bool = False) -> Path:
    names = [(f"{label}{i}.jpg", label) for label in labels for i in range(per_label)]
    rows = "id,file,label\n" + "".join(f"r{k},{name},{label}\n" for k, (name, label) in enumerate(names))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("labels.csv", ("﻿" if bom else "") + rows)
        for k, (name, _label) in enumerate(names):
            if name != drop:
                archive.writestr(name, _jpeg(k, corrupt=name == corrupt))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return path


def test_vit_m1_stated_minimum_is_what_the_split_accepts(tmp_path, notebook):
    assert sm.min_byod_records(2)["total"] == 12 and sm.min_byod_records(3)["total"] == 15 and sm.min_byod_records(4)["total"] == 16
    split = sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "ok.zip", 6)), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"test": 2, "validation": 2, "train": 8}
    with pytest.raises(ValueError, match=r"supply at least 12 distinct photographs, 6 per label"):
        sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "small.zip", 5)), seed=42)
    assert "**12 photographs for two labels**" in _markdown(notebook)


def test_vit_m1_missing_and_corrupt_images_name_their_row_and_litter_is_skipped(tmp_path):
    with pytest.raises(ValueError, match=r"labels.csv line 5 \(file 'cat3.jpg'\): that image file is not in the dataset"):
        sm.load_byod_dataset(_zip(tmp_path / "missing.zip", 6, drop="cat3.jpg"))
    with pytest.raises(ValueError, match=r"labels.csv line 3 \(file 'cat1.jpg'\): Pillow cannot decode the image"):
        sm.load_byod_dataset(_zip(tmp_path / "corrupt.zip", 6, corrupt="cat1.jpg"))
    assert len(sm.load_byod_dataset(_zip(tmp_path / "mac.zip", 6, extra={"__MACOSX/._cat0.jpg": b"\0"}, bom=True))) == 12


def _section_4(notebook: dict, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace(restored: list) -> dict:
    from vit_classification_pipeline import metrics as mt
    from vit_classification_pipeline import pipeline as pl

    ns = {}
    for module in (pl, mt, sm):
        ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("__")})
    pipe = types.SimpleNamespace(adapter={"policy": "x"}, restore_base=lambda: restored.append(True) or ["a"])
    ns.update({"os": __import__("os"), "Path": Path, "pipe": pipe, "__name__": "__main__"})
    return ns


def test_vit_m1_byod_path_runs_section_4_outside_colab_from_the_base(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _zip(tmp_path / "mine.zip", 7)
    restored: list = []
    ns = _section_4_namespace(restored)
    exec(_section_4(notebook, "mine.zip"), ns)
    out = capsys.readouterr().out
    assert restored == [True], "a BYOD re-run must put the model back to the pinned base first"
    assert ns["raw_count"] == {"byod": 14, "labels": 2, "duplicate_images_dropped": 0, "effective_minimum": 12}
    assert "fewer than 5 per label" in out
    assert (tmp_path / "outputs" / f"{STEM}_train.csv").is_file()


def test_vit_m1_upload_outside_colab_cancelled_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, ""), _section_4_namespace([]))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.zip' does not exist"):
        exec(_section_4(notebook, "nowhere.zip"), _section_4_namespace([]))
    for uploaded, message in (({}, "received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2")):
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, ""), _section_4_namespace([]))


# --- VIT-m2: quality outcomes are reported verdicts -------------------------------------------------------------


def test_vit_m2_no_quality_assert_remains(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = re.findall(r"(?m)^\s*assert .*$", code)
    assert len(asserts) == 1 and asserts[0].startswith("assert parity['probabilities_identical']")
    assert not re.search(r"(?m)^assert .*floor", code)


def _m(accuracy):
    return {"accuracy": accuracy, "macro_f1": accuracy, "n": 4, "log_loss": 0.7, "verdict": "measured-small-sample", "definitions": {}, "per_class": {c: {"recall": accuracy} for c in ("cat", "dog")}, "confusion": [[1, 1], [1, 1]], "baseline": "stand-in"}


def test_vit_m2_a_result_at_the_floor_is_recorded_and_does_not_stop_the_notebook(notebook, tmp_path, monkeypatch):
    """Sections 6 and 8 executed with stand-ins at the majority floor (a small BYOD test split): both complete and
    record the verdicts (stand-in evidence, no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scores = iter([_m(0.5), _m(0.25), _m(0.5)])

    class StandIn:
        def knn_baseline(self, *a, **k):
            return _m(0.5)

        def adapt(self, *a, **k):
            return {"policy": "frozen backbone + linear probe", "probe_final_loss": 0.1, "history": [{"val": {"accuracy": 0.5}}], "trainable_names": []}

        def evaluate(self, records):
            return next(scores)

    ns = {
        "pipe": StandIn(), "train_records": [{"label": "cat"}], "val_records": [], "test_records": [{"label": "cat"}, {"label": "dog"}], "time": __import__("time"), "json": json,
        "majority_baseline": lambda *a: _m(0.5), "classes": ["cat", "dog"],
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "MODEL_KEY": "stand-in", "data_source": "stand-in", "dataset_manifests": {"test": {"digest": "d"}},
        "disjoint": {}, "overlap": {}, "adapt_result": {"policy": "frozen backbone + linear probe", "history": []}, "adapt_seconds": 0.0,
    }
    exec(_cell(notebook, "floor = majority_baseline("), ns)
    assert ns["frozen_verdict"] == "at or below floor"
    exec(_cell(notebook, "adapted_test = pipe.evaluate(test_records)"), ns)
    verdicts = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))["comparison"]["verdicts"]
    assert verdicts == {"frozen_vs_floor": "at or below floor", "selected_vs_floor": "at or below floor", "selected_vs_frozen_accuracy": "worse"}


def test_vit_m2_policy_identity_stays_a_hard_check(notebook):
    source = _cell(notebook, "floor = majority_baseline(")
    assert "if not probe_result['policy'].startswith('frozen'):" in source and "raise RuntimeError" in source


# --- VIT-m3: no doubled braces --------------------------------------------------------------------------------


def test_vit_m3_no_escaped_braces_in_the_data_contract(notebook):
    markdown = _markdown(notebook)
    assert "{{" not in markdown and "}}" not in markdown
    assert "`[A-Za-z0-9_.:-]{1,64}`" in markdown and "`{id, image, label}`" in markdown

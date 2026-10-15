"""Checks of the 2026-10-15 ICR2 review fixes to the multi-model image-classification workshop.

They run the notebook's own cells with inert stand-ins (fake predictions, tiny synthetic images); no
backbone is downloaded or run, so none of this is pretrained-model evidence.
"""
import ast
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads(
    (ROOT / "tutorials/DIMER_MultiModel_Image_Classification_Workshop.ipynb").read_text(encoding="utf-8")
)
BY_ID = {c["id"]: "".join(c["source"]) for c in NB["cells"]}
MARKDOWN = "\n".join("".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "markdown")
SPECIES_SYNSETS = {
    "n01532325": "American goldfinch",
    "n01532829": "house finch",
    "n01534582": "dark-eyed junco",
    "n01535469": "white-throated sparrow",
    "n01536035": "chipping sparrow",
    "n01536644": "song sparrow",
}


def numbered(n):
    return BY_ID[f"dimer-image-workshop-{n:02d}"]


def controls_and_specs(tmp_path, monkeypatch, **overrides):
    monkeypatch.chdir(tmp_path)
    ns = {"display": lambda *a, **k: None}
    exec(numbered(4), ns)
    ns.update(overrides)
    if overrides:
        ns["SELECTED_MODELS"] = (
            ns["CORE_MODELS"]
            + (ns["IN1K_COUNTERPARTS"] if ns["RUN_IN1K_COUNTERPARTS"] else [])
            + (["dinov2"] if ns["RUN_DINOV2_EXTENSION"] else [])
        )
    exec(numbered(7), ns)
    return ns


def functions(cell_source, *names):
    namespace = {"math": math}
    for node in ast.parse(cell_source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), node.name, "exec"), namespace)
    return [namespace[name] for name in names]


# --- ICR2-M1: pretraining labels are disclosed, and the in1k-only counterparts run by default ------------

def test_counterparts_run_by_default_and_keep_the_core_set(tmp_path, monkeypatch):
    ns = controls_and_specs(tmp_path, monkeypatch)
    assert ns["CORE_MODELS"] == ["mobilenet", "resnet", "convnext", "vit"]
    assert ns["SELECTED_MODELS"] == ["mobilenet", "resnet", "convnext", "vit", "convnext_in1k", "vit_in1k"]
    assert ns["RUN_CONTROLS"]["in1k_counterparts"] is True
    groups = [ns["model_group"](k) for k in ns["SELECTED_MODELS"]]
    assert groups == ["core"] * 4 + ["ImageNet-1k-only counterpart"] * 2
    assert ns["model_group"]("dinov2") == "self-supervised baseline"


def test_counterparts_are_pinned_like_the_core_models(tmp_path, monkeypatch):
    specs = controls_and_specs(tmp_path, monkeypatch)["MODEL_SPECS"]
    for key, base in (("convnext_in1k", "convnext"), ("vit_in1k", "vit")):
        spec = specs[key]
        assert spec["model_id"].split(".")[0] == specs[base]["model_id"].split(".")[0]
        assert spec["model_id"].endswith("_in1k")
        assert "in12k" not in spec["model_id"] and "in21k" not in spec["model_id"]
        assert len(spec["revision"]) == 40 and int(spec["revision"], 16) >= 0
        assert spec["native_input"] == specs[base]["native_input"] and spec["env"] == "common"
        assert set(spec["files"]) == {"README.md", "config.json", "model.safetensors"}
        assert all(len(digest) == 64 for _, digest in spec["files"].values())
    weights = [specs[k]["files"]["model.safetensors"][0] for k in ("convnext_in1k", "convnext")]
    assert weights[0] == weights[1]


def test_every_checkpoint_has_a_pretraining_label_disclosure(tmp_path, monkeypatch):
    ns = controls_and_specs(tmp_path, monkeypatch)
    labels = ns["PRETRAINING_LABELS"]
    assert set(labels) == set(ns["MODEL_SPECS"])
    for key in ("convnext", "vit", "eva"):
        assert labels[key][1] == "all six"
    for key in ("mobilenet", "resnet", "convnext_in1k", "vit_in1k", "swin"):
        assert labels[key][0] == "ImageNet-1k" and labels[key][1].startswith("house finch only")


def test_disclosed_coverage_matches_timms_synset_lists():
    spec = importlib.util.find_spec("timm")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("timm is not installed")
    info = Path(list(spec.submodule_search_locations)[0]) / "data" / "_info"
    sets = {
        name: set((info / f"{name}_synsets.txt").read_text().split())
        for name in ("imagenet", "imagenet12k", "imagenet22k", "imagenet21k_goog")
    }
    for synset in SPECIES_SYNSETS:
        assert all(synset in sets[name] for name in ("imagenet12k", "imagenet22k", "imagenet21k_goog"))
        assert synset in MARKDOWN
    assert {s for s in SPECIES_SYNSETS if s in sets["imagenet"]} == {"n01532829"}  # house finch only
    assert "n01534433" in sets["imagenet"] and "n01534433" in MARKDOWN  # the broader "junco, snowbird" class


def test_framing_no_longer_credits_architecture():
    opening = "".join(NB["cells"][0]["source"])
    assert "Controlled comparison" not in opening and "Core architectures" not in opening
    assert "cannot be credited to architecture or model size" in opening
    assert "CNN/transformer disagreements" not in MARKDOWN
    assert "architecture-specific successes" not in MARKDOWN
    assert "What visual characteristics might explain the differing representations?" not in MARKDOWN


def test_runner_always_uses_the_state_dict_loader():
    runner = numbered(20)
    assert runner.count('overlay["custom_load"]=False') == 1
    assert runner.index('overlay["custom_load"]=False') < runner.index('overlay["file"]=')


# --- ICR2-m3: nominal Wilson intervals and paired exact McNemar tables -----------------------------------

def test_wilson_and_mcnemar_match_closed_forms():
    wilson, mcnemar = functions(BY_ID["icr2-matched-in1k"], "wilson_interval", "exact_mcnemar_p")
    lo, hi = wilson(24, 48)
    # By hand: centre 0.5, half-width 0.13611.
    assert lo == pytest.approx(0.36389, abs=1e-5) and hi == pytest.approx(0.63611, abs=1e-5)
    assert wilson(48, 48)[1] == pytest.approx(1.0) and wilson(0, 48)[0] == pytest.approx(0.0, abs=1e-12)
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(0, 5) == pytest.approx(2 / 32)
    assert mcnemar(3, 6) == pytest.approx(2 * sum(math.comb(9, i) for i in range(4)) / 2**9)
    with pytest.raises(ValueError):
        wilson(49, 48)


def scored_stand_in(tmp_path, monkeypatch, correct_by_key):
    ns = controls_and_specs(tmp_path, monkeypatch)
    ids = [f"test-{i:03d}" for i in range(len(next(iter(correct_by_key.values()))))]
    truth = ["a"] * len(ids)
    ns["expected_split"] = lambda split: pd.DataFrame({"id": ids, "label": truth})
    ns["test_predictions"], ns["test_metrics"] = {}, {}
    for key, correct in correct_by_key.items():
        pred = ["a" if ok else "b" for ok in correct]
        ns["test_predictions"][key] = pd.DataFrame({"id": ids, "truth": truth, "prediction": pred})
        ns["test_metrics"][key] = {"accuracy": float(np.mean(correct)), "macro_f1": 0.5, "log_loss": 1.0}
    ns["SELECTED_MODELS"] = list(correct_by_key)
    ns["OUTPUT_ROOT"] = tmp_path / "out"
    (ns["OUTPUT_ROOT"] / "test").mkdir(parents=True)
    return ns


def test_matched_and_paired_tables_on_stand_in_predictions(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    keys = ["mobilenet", "resnet", "convnext", "vit", "convnext_in1k", "vit_in1k"]
    correct = {k: rng.random(48) < 0.6 for k in keys}
    ns = scored_stand_in(tmp_path, monkeypatch, correct)
    exec(BY_ID["icr2-matched-in1k"], ns)
    exec(BY_ID["icr2-paired-comparison"], ns)
    out = capsys.readouterr().out
    matched = ns["matched_in1k_table"]
    assert list(matched["model"]) == [ns["MODEL_SPECS"][k]["display_name"] for k in
                                      ("mobilenet", "resnet", "convnext_in1k", "vit_in1k")]
    assert "3. Each core checkpoint against its ImageNet-1k-only counterpart" in out
    paired = ns["paired_table"]
    assert len(paired) == 15
    row = paired[(paired["model A"] == "ConvNeXt-Tiny") & (paired["model B"] == "ViT-B/16")].iloc[0]
    assert row["only A right"] == int((correct["convnext"] & ~correct["vit"]).sum())
    assert row["only B right"] == int((~correct["convnext"] & correct["vit"]).sum())
    assert (tmp_path / "out/test/paired_comparisons.csv").is_file()
    assert "Bonferroni" in out


def test_intervals_refuse_a_count_that_disagrees_with_the_metric(tmp_path, monkeypatch):
    ns = scored_stand_in(tmp_path, monkeypatch, {"mobilenet": np.array([True, False, True, True])})
    ns["test_metrics"]["mobilenet"]["accuracy"] = 0.5
    with pytest.raises(RuntimeError, match="disagrees with the recorded test accuracy"):
        exec(BY_ID["icr2-matched-in1k"], ns)


# --- ICR2-m4: BYOD refusals are one message, without a traceback ----------------------------------------

def byod_dataset(root, padded=True, missing=True, undecodable=False):
    root.mkdir()
    rows = ["id,file,label"]
    for i in range(16):
        Image.new("RGB", (8, 8), (i, i * 3, i * 7)).save(root / f"{i}.png")
        rows.append(f"{i},{i}.png,{'a' if i < 8 else 'b'}")
    if padded:
        rows[3] = "2,2.png, a"
    if missing:
        rows[5] = "4,absent.png,a"
    if undecodable:
        (root / "6.png").write_bytes(b"not an image")
    (root / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def acquire(tmp_path, monkeypatch, path):
    ns = controls_and_specs(tmp_path, monkeypatch, USE_BYOD=True, BYOD_PATH=str(path))
    ns["splits"] = {"stale": []}  # an earlier acquisition's data must not survive a refusal
    exec(numbered(11), ns)
    return ns


def test_byod_refusal_names_every_problem_once_without_a_traceback(tmp_path, monkeypatch, capsys):
    path = byod_dataset(tmp_path / "input")
    with pytest.raises(ValueError) as caught:
        acquire(tmp_path, monkeypatch, path)
    refusal = caught.value
    assert type(refusal).__name__ == "ByodRefused"
    assert refusal._render_traceback_() == []  # IPython prints no traceback for it
    out = capsys.readouterr().out
    assert out.count("BYOD refused:") == 1
    assert "whitespace" in out and "missing BYOD image: absent.png" in out
    assert "line 4" in out and "line 6" in out
    assert "run this cell again" in out


def test_byod_refusal_leaves_no_stale_data(tmp_path, monkeypatch):
    path = byod_dataset(tmp_path / "input")
    monkeypatch.chdir(tmp_path)
    ns = controls_and_specs(tmp_path, monkeypatch, USE_BYOD=True, BYOD_PATH=str(path))
    ns["splits"] = {"stale": []}
    with pytest.raises(ValueError):
        exec(numbered(11), ns)
    assert "splits" not in ns and "records" not in ns


def test_byod_too_few_images_lists_each_small_class(tmp_path, monkeypatch, capsys):
    root = tmp_path / "input"
    root.mkdir()
    rows = ["id,file,label"]
    for i in range(12):
        Image.new("RGB", (8, 8), (i, 2 * i, 3 * i)).save(root / f"{i}.png")
        rows.append(f"{i},{i}.png,{'abc'[i % 3]}")
    (root / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 8 images"):
        acquire(tmp_path, monkeypatch, root)
    out = capsys.readouterr().out
    assert all(f"class '{label}': 4 image(s)" in out for label in "abc")


def test_a_valid_byod_dataset_still_splits_identically(tmp_path, monkeypatch):
    ns = acquire(tmp_path, monkeypatch, byod_dataset(tmp_path / "input", padded=False, missing=False))
    assert {k: len(v) for k, v in ns["splits"].items()} == {"train": 10, "validation": 2, "test": 4}


def test_an_undecodable_image_is_an_error_not_a_refusal(tmp_path, monkeypatch):
    path = byod_dataset(tmp_path / "input", padded=False, missing=False, undecodable=True)
    with pytest.raises(OSError):
        acquire(tmp_path, monkeypatch, path)


# --- ICR2-m1, m2, m5, m6, S1 ------------------------------------------------------------------------------

def test_disclosures_and_declarations():
    assert "cannot be ruled out" in numbered(2)
    assert "not tuned per model" in numbered(23)
    section10 = numbered(23)
    assert "b9b94c9" in section10 and "5874024a" in section10 and "not re-run here" in section10
    assert NB["metadata"]["dimer"]["notebook_spec"] == "2.5"
    assert NB["metadata"]["dimer"]["notebook_kind"] == "WORKSHOP"
    assert "**Notebook specification:** `2.5`" in "".join(NB["cells"][0]["source"])
    assert "No confidence interval or general superiority claim is justified" not in MARKDOWN


def test_majority_baseline_reports_a_tie(capsys):
    ns = {"np": np, "class_names": ["a", "b"], "splits": {
        "train": [{"label": x} for x in "abab"], "validation": [{"label": x} for x in "ab"]}}
    exec(numbered(16), ns)
    assert "2 classes tie at 2 training images" in capsys.readouterr().out

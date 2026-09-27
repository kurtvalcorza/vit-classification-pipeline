"""Acceptance checks for the Notebook Review Framework v1 image-classification findings (ICR-01 … ICR-08).

The tests execute the workshop notebook's own cells end to end on a small BYOD fixture. The common probe
training and inference programs are the notebook's real runners (PyTorch + SafeTensors, CPU). Only the
pretrained-backbone feature extractor is replaced, by a fake program with the same command-line and file
contract that writes deterministic synthetic features. These are orchestration and contract checks, not
backbone runs or Colab evidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/DIMER_MultiModel_Image_Classification_Workshop.ipynb"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {
    "".join(c["source"]).splitlines()[0].removeprefix("# @title "): "".join(c["source"])
    for c in NB["cells"]
    if c["cell_type"] == "code"
}
MARKDOWN = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "markdown"]

CORE_STANDARD_LINE = 'CORE_STANDARD = ["mobilenet", "resnet", "convnext", "vit"]'
ENVIRONMENT_BUILD = (
    'ensure_uv()\nneeded_envs = sorted({MODEL_SPECS[m]["env"] for m in SELECTED_MODELS})\n'
    "ENV_PYTHONS = {name: ensure_env(name) for name in needed_envs}"
)
FAKE_ENVIRONMENTS = """
needed_envs = sorted({MODEL_SPECS[m]["env"] for m in SELECTED_MODELS})
for name in needed_envs:
    (WORK_ROOT / "envs" / name).mkdir(parents=True, exist_ok=True)
    (WORK_ROOT / "envs" / name / ".dimer_ready").write_text("signature-" + name)
ENV_PYTHONS = {name: Path(sys.executable) for name in needed_envs}
"""

FAKE_FEATURE_RUNNER = r"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
for flag in ("--spec", "--dataset", "--features", "--metadata", "--cache"):
    ap.add_argument(flag, required=True)
args = ap.parse_args()
spec = json.loads(Path(args.spec).read_text())
with open(args.dataset, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
features = []


def seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


for row in rows:
    # A class-specific centre plus a small image-specific offset: learnable, deterministic.
    centre = np.random.default_rng(seed(row["label"])).normal(size=16)
    offset = np.random.default_rng(seed(row["id"])).normal(size=16)
    features.append(centre * 3 + offset * 0.5)
np.savez_compressed(
    args.features,
    features=np.asarray(features, dtype=np.float32),
    ids=np.asarray([r["id"] for r in rows], dtype=str),
    labels=np.asarray([r["label"] for r in rows], dtype=str),
    splits=np.asarray([r["split"] for r in rows], dtype=str),
)
Path(args.metadata).write_text(json.dumps({
    "key": spec["key"], "display_name": spec["display_name"], "family": spec["family"],
    "model_id": spec["model_id"], "revision": spec["revision"], "license": spec["license"],
    "device": "cpu", "parameters": 1000, "feature_dimension": 16, "n_images": len(rows),
    "data_config": {"input_size": [3, 8, 8]}, "native_input": spec["native_input"],
    "load_seconds": 0.0, "feature_seconds": 0.01, "images_per_second": 100.0,
    "peak_cuda_bytes": None, "python": "test", "timm": "fake", "torch": "fake",
    "weight_bytes": int(spec["files"]["model.safetensors"][0]),
    "weight_sha256": spec["files"]["model.safetensors"][1],
}))
"""


class Axis:
    def __init__(self, log):
        self.log = log

    def set_title(self, text, *args, **kwargs):
        self.log.append(("title", str(text)))

    def text(self, x, y, text, *args, **kwargs):
        self.log.append(("text", str(text)))

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class PyplotStub:
    """Records gallery captions; everything else is a no-op."""

    def __init__(self):
        self.log = []

    def subplots(self, *shape, squeeze=True, **kwargs):
        if not shape:
            return self, Axis(self.log)
        rows, cols = (shape + (1,))[:2]
        grid = np.empty((rows, cols), dtype=object)
        for index in np.ndindex(rows, cols):
            grid[index] = Axis(self.log)
        if squeeze and grid.size == 1:
            return self, grid[0, 0]
        return self, grid if not squeeze else grid.squeeze()

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def plots(monkeypatch):
    stub = PyplotStub()
    package = types.ModuleType("matplotlib")
    package.pyplot = stub
    monkeypatch.setitem(sys.modules, "matplotlib", package)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", stub)
    import importlib.metadata

    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "3.10.0" if name == "matplotlib" else real_version(name)
    )
    return stub


def byod_fixture(root, labels, per_class=10):
    root.mkdir(parents=True)
    rows, index = [], 0
    for label in labels:
        for _ in range(per_class):
            Image.new("RGB", (8, 8), (index % 256, (index * 3) % 256, (index * 7) % 256)).save(
                root / f"{index}.png"
            )
            rows.append({"id": f"img{index}", "file": f"{index}.png", "label": label})
            index += 1
    with open(root / "labels.csv", "w", encoding="utf-8", newline="") as handle:
        handle.write("id,file,label\n")
        for row in rows:
            handle.write(f'{row["id"]},{row["file"]},"{row["label"]}"\n')
    return root


def run_cell(title, ns, replacements=None):
    source = CELLS[title]
    for old, new in (replacements or {}).items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, title, "exec"), ns)
    return ns


def start(tmp_path, monkeypatch, labels=("alpha", "beta", "gamma"), steps=20, seed=0, name="input"):
    """Controls through the validation evaluator, for a BYOD fixture and two core models."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / name
    if not data.exists():
        byod_fixture(data, labels)
    ns = {"display": lambda *a, **k: None}
    run_cell(
        "Notebook controls",
        ns,
        {
            "USE_BYOD = False  ": "USE_BYOD = True   ",
            'BYOD_PATH = ""    ': f'BYOD_PATH = "{data}"  ',
            "PROBE_STEPS = 1000 ": f"PROBE_STEPS = {steps} ",
            "PROBE_SEED = 0 ": f"PROBE_SEED = {seed} ",
            CORE_STANDARD_LINE: 'CORE_STANDARD = ["mobilenet", "vit"]',
        },
    )
    for title in [
        "Verify the orchestration runtime",
        "MODEL_SPECS = {",
        "Acquire and validate built-in data or BYOD",
        "Export the validated dataset manifest",
        "Class counts and representative image gallery",
        "Majority baseline",
    ]:
        run_cell(title, ns)
    run_cell("Create isolated pinned environments", ns, {ENVIRONMENT_BUILD: FAKE_ENVIRONMENTS})
    run_cell("Write the generic feature-extraction runner", ns)
    ns["feature_runner_path"].write_text(FAKE_FEATURE_RUNNER, encoding="utf-8")
    ns["RUNNER_SHA256"]["features"] = ns["sha256_file"](ns["feature_runner_path"])
    for title in [
        "Run feature extraction one model at a time",
        "Write the common probe-training runner",
        "Train the common probe and verify serialized reload on validation",
        "Common validation evaluator",
    ]:
        run_cell(title, ns)
    return ns


def freeze(ns):
    return run_cell("Freeze experiment", ns)


def finish(ns):
    for title in [
        "Reload each frozen adapter and score test features",
        "Confusion matrices",
        "Cross-model disagreement table",
        "Deterministic error gallery",
        "Architecture / compute comparison",
        "Compare fresh-reloaded predictions for selected held-out images",
        "Write final provenance",
        "Run-all completion summary",
    ]:
        run_cell(title, ns)
    return ns


# --- ICR-01: class labels survive every file boundary unchanged ---------------------------------------

LONG_A = "é" * 129 + "a"
LONG_B = "é" * 129 + "b"


@pytest.mark.parametrize(
    "labels",
    [
        ("alpha", "beta", "gamma"),
        ("0", "1", "2"),
        ("001", "1", "002"),
        ("NA", "null", "other"),
        (LONG_A, LONG_B, "short"),
    ],
    ids=["alphabetic", "numeric", "leading-zero", "missing-tokens", "long-unicode"],
)
def test_labels_are_kept_exactly_from_acquisition_to_export(tmp_path, monkeypatch, plots, labels):
    ns = finish(freeze(start(tmp_path, monkeypatch, labels)))
    assert ns["class_names"] == sorted(labels)
    for key in ns["SELECTED_MODELS"]:
        manifest = json.loads((ns["OUTPUT_ROOT"] / "artifacts" / key / "manifest.json").read_text())
        assert manifest["class_order"] == sorted(labels)
        with np.load(ns["feature_files"][key], allow_pickle=False) as data:
            assert set(data["labels"].astype(str)) == set(labels)
        assert set(ns["test_predictions"][key]["truth"]) == set(labels)
    exported = pd.read_csv(ns["OUTPUT_ROOT"] / "test/mobilenet.csv", dtype=str, keep_default_na=False)
    assert set(exported["truth"]) == set(labels)


@pytest.mark.parametrize(
    ("label", "message"), [(" padded", "whitespace"), ("two\tparts", "control character")]
)
def test_ambiguous_labels_are_refused_before_any_model(tmp_path, monkeypatch, plots, label, message):
    with pytest.raises(ValueError, match=message):
        start(tmp_path, monkeypatch, labels=("alpha", label))
    assert not list(tmp_path.rglob("*.npz"))


# --- ICR-02: every model is scored on the same complete, authoritative test set ------------------------


@pytest.fixture
def scored(tmp_path, monkeypatch, plots):
    return finish(freeze(start(tmp_path, monkeypatch)))


def _variant(ns, kind):
    path = ns["test_prediction_files"]["mobilenet"]
    pred = pd.read_csv(path, dtype=str, keep_default_na=False)
    probs = [c for c in pred.columns if c.startswith("prob_")]
    if kind == "missing_row":
        pred = pred.iloc[1:]
    elif kind == "duplicate_row":
        pred.iloc[0] = pred.iloc[1]
    elif kind == "extra_row":
        extra = pred.iloc[[0]].copy()
        extra["id"] = "not-in-the-dataset"
        pred = pd.concat([pred, extra])
    elif kind == "changed_truth":
        pred.loc[0, "truth"] = next(c for c in ns["class_names"] if c != pred.loc[0, "truth"])
    elif kind == "probabilities_above_one":
        pred[probs] = "2.0"
    elif kind == "not_normalised":
        pred[probs] = "0.1"
    elif kind == "prediction_not_argmax":
        top = pd.to_numeric(pred.loc[0, probs]).idxmax().removeprefix("prob_")
        pred.loc[0, "prediction"] = next(c for c in ns["class_names"] if c != top)
    elif kind == "missing_probability":
        pred.loc[0, probs[0]] = ""
    elif kind == "class_order":
        pred = pred[["id", "truth", "prediction", *reversed(probs)]]
    out = path.with_name(f"variant-{kind}.csv")
    pred.to_csv(out, index=False)
    return out


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing_row", "1 expected image\\(s\\) missing"),
        ("duplicate_row", "duplicate image IDs"),
        ("extra_row", "1 unexpected"),
        ("changed_truth", "differs from the dataset"),
        ("probabilities_above_one", "outside \\[0, 1\\]"),
        ("not_normalised", "do not sum to 1"),
        ("prediction_not_argmax", "not the highest-probability class"),
        ("missing_probability", "missing or non-finite"),
        ("class_order", "expected class order"),
    ],
)
def test_invalid_prediction_files_are_rejected_before_metrics(scored, kind, message):
    with pytest.raises(ValueError, match=message):
        scored["evaluate_prediction_csv"](
            _variant(scored, kind), scored["class_names"], scored["expected_split"]("test")
        )


def test_a_permuted_file_realigns_by_id_and_keeps_its_scores(scored):
    path = scored["test_prediction_files"]["mobilenet"]
    shuffled = pd.read_csv(path, dtype=str, keep_default_na=False).sample(frac=1, random_state=1)
    out = path.with_name("shuffled.csv")
    shuffled.to_csv(out, index=False)
    _, metrics = scored["evaluate_prediction_csv"](
        out, scored["class_names"], scored["expected_split"]("test")
    )
    reference = scored["test_metrics"]["mobilenet"]
    assert metrics["accuracy"] == reference["accuracy"]
    assert metrics["log_loss"] == pytest.approx(reference["log_loss"])
    assert metrics["n_images"] == len(scored["splits"]["test"])


def test_disagreement_table_covers_every_test_image(scored):
    assert len(scored["base"]) == len(scored["splits"]["test"])
    assert set(scored["base"]["id"]) == {r["id"] for r in scored["splits"]["test"]}


# --- ICR-03: the freeze binds every file that decides a test prediction --------------------------------


def _reverse_class_order(ns):
    path = ns["OUTPUT_ROOT"] / "artifacts/mobilenet/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["class_order"] = list(reversed(manifest["class_order"]))
    path.write_text(json.dumps(manifest, indent=2))


def _change_features(ns):
    path = ns["feature_files"]["mobilenet"]
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["features"] = arrays["features"][::-1].copy()
    np.savez_compressed(path, **arrays)


def _change_split(ns):
    path = ns["feature_files"]["vit"]
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["splits"] = np.where(arrays["splits"] == "test", "train", arrays["splits"])
    np.savez_compressed(path, **arrays)


MUTATIONS = {
    "class_order": _reverse_class_order,
    "features": _change_features,
    "split_membership": _change_split,
    "adapter": lambda ns: (ns["OUTPUT_ROOT"] / "artifacts/vit/adapter.safetensors").open("ab").write(b"\0"),
    "infer_runner": lambda ns: ns["probe_infer_path"].write_text(
        ns["probe_infer_path"].read_text() + "\n# edit\n"
    ),
    "environment": lambda ns: (ns["WORK_ROOT"] / "envs/common/.dimer_ready").write_text("rebuilt"),
    "probe_setting": lambda ns: ns.update(PROBE_SEED=7),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_any_change_after_the_freeze_stops_scoring(tmp_path, monkeypatch, plots, mutation):
    ns = freeze(start(tmp_path, monkeypatch))
    MUTATIONS[mutation](ns)
    calls = []
    ns["run_checked"] = lambda cmd, label: calls.append(label)
    with pytest.raises(RuntimeError, match="no longer matches the freeze"):
        run_cell("Reload each frozen adapter and score test features", ns)
    assert calls == []


def test_inference_program_checks_digests_itself(tmp_path, monkeypatch, plots):
    ns = freeze(start(tmp_path, monkeypatch))
    frozen = json.loads(ns["freeze_path"].read_text())["models"]["mobilenet"]
    _reverse_class_order(ns)
    completed = subprocess.run(
        [
            sys.executable,
            str(ns["probe_infer_path"]),
            "--features",
            str(ns["feature_files"]["mobilenet"]),
            "--artifact-dir",
            str(ns["OUTPUT_ROOT"] / "artifacts/mobilenet"),
            "--split",
            "test",
            "--predictions",
            str(tmp_path / "out.csv"),
            "--expected-features-sha256",
            frozen["features_sha256"],
            "--expected-manifest-sha256",
            frozen["probe_manifest_sha256"],
            "--expected-adapter-sha256",
            frozen["probe_artifact_sha256"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "manifest.json differs from the frozen experiment" in completed.stderr
    assert not (tmp_path / "out.csv").exists()


def test_freeze_is_kept_and_the_test_is_scored_once(tmp_path, monkeypatch, plots):
    ns = freeze(start(tmp_path, monkeypatch))
    original = ns["freeze_path"].read_bytes()
    freeze(ns)
    assert ns["freeze_path"].read_bytes() == original
    ns["PROBE_SEED"] = 3
    with pytest.raises(RuntimeError, match="already frozen"):
        freeze(ns)
    ns["PROBE_SEED"] = 0
    run_cell("Reload each frozen adapter and score test features", ns)
    with pytest.raises(RuntimeError, match="already been scored"):
        run_cell("Reload each frozen adapter and score test features", ns)


# --- ICR-04: readable, de-duplicated error gallery ----------------------------------------------------


def test_error_gallery_captions_are_one_model_per_short_line(scored, plots):
    plots.log.clear()
    run_cell("Deterministic error gallery", scored)
    captions = [text for kind, text in plots.log]
    assert captions
    assert all(len(line) <= 40 for text in captions for line in text.splitlines())
    model_lines = [line for kind, text in plots.log if kind == "text" for line in text.splitlines()]
    for key in scored["CORE_MODELS"]:
        assert sum(scored["MODEL_SPECS"][key]["display_name"] in line for line in model_lines) == len(
            scored["gallery_ids"]
        )
    ids = [rid for _, rid in scored["gallery_ids"]]
    assert len(ids) == len(set(ids))


def test_long_labels_are_wrapped_in_the_gallery(tmp_path, monkeypatch, plots):
    ns = finish(freeze(start(tmp_path, monkeypatch, (LONG_A, LONG_B, "short"))))
    plots.log.clear()
    run_cell("Class counts and representative image gallery", ns)
    run_cell("Deterministic error gallery", ns)
    assert plots.log
    assert all(len(line) <= 40 for _, text in plots.log for line in text.splitlines())
    assert ns["gallery_ids"]


# --- ICR-05: the BYOD contract precedes the upload -------------------------------------------------------


def test_byod_contract_and_privacy_warning_precede_the_acquisition_cell():
    order = ["".join(c["source"]) for c in NB["cells"]]
    acquisition = next(i for i, s in enumerate(order) if s.startswith("# @title Acquire and validate"))
    before = "\n".join(order[:acquisition])
    for literal in [
        "2 to 100 classes",
        "8 images per class",
        "1–4096 pixels",
        "2 GiB",
        "**Privacy.**",
        "What leaves the runtime in the report",
        "kept **exactly**",
    ]:
        assert literal in before


# --- ICR-06: each report contains only its own experiment ----------------------------------------------


def test_each_report_contains_only_its_own_experiment(tmp_path, monkeypatch, plots):
    first = finish(freeze(start(tmp_path, monkeypatch)))
    first_report = Path(first["bundle"])
    first_bytes = first_report.read_bytes()
    extra = first["OUTPUT_ROOT"] / "test" / "eva.csv"
    extra.write_text("stale file from a FULL run")

    second = finish(freeze(start(tmp_path, monkeypatch, steps=30)))
    assert second["OUTPUT_ROOT"] != first["OUTPUT_ROOT"]
    with zipfile.ZipFile(second["bundle"]) as archive:
        names = set(archive.namelist())
        inventory = json.loads(archive.read("provenance/inventory.json"))
        summary = json.loads(archive.read("workshop_summary.json"))
    assert "test/eva.csv" not in names
    assert names == {item["path"] for item in inventory["files"]} | {"provenance/inventory.json"}
    assert summary["experiment_id"] == second["EXPERIMENT_ID"]
    assert first_report.read_bytes() == first_bytes
    assert (first["OUTPUT_ROOT"] / "artifacts/mobilenet/adapter.safetensors").exists()


# --- ICR-07: a validation-only handoff for the step-budget activity ------------------------------------


def test_activity_records_compare_only_a_single_change(tmp_path, monkeypatch, plots):
    reference = start(tmp_path, monkeypatch, steps=20)
    changed = start(tmp_path, monkeypatch, steps=10)
    for ns in (reference, changed):
        record = json.loads(ns["ACTIVITY_RECORD_PATH"].read_text())
        assert "test" not in json.dumps(record).lower()
    run_cell(
        "Compare two saved step-budget records (optional)",
        changed,
        {
            'REFERENCE_RECORD = ""': f'REFERENCE_RECORD = "{reference["ACTIVITY_RECORD_PATH"]}"',
            'CHANGED_RECORD = ""': f'CHANGED_RECORD = "{changed["ACTIVITY_RECORD_PATH"]}"',
        },
    )
    table = changed["step_budget_comparison"]
    assert list(table["model"]) == ["MobileNetV4-Conv-Small", "ViT-B/16"]
    assert "difference_changed_minus_reference" in table

    other_seed = start(tmp_path, monkeypatch, steps=10, seed=5)
    with pytest.raises(ValueError, match="differ only in probe_steps"):
        run_cell(
            "Compare two saved step-budget records (optional)",
            other_seed,
            {
                'REFERENCE_RECORD = ""': f'REFERENCE_RECORD = "{reference["ACTIVITY_RECORD_PATH"]}"',
                'CHANGED_RECORD = ""': f'CHANGED_RECORD = "{other_seed["ACTIVITY_RECORD_PATH"]}"',
            },
        )

    tampered = json.loads(reference["ACTIVITY_RECORD_PATH"].read_text())
    tampered["models"]["vit"]["log_loss"] = 0.0
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="not an unmodified validation record"):
        changed["load_activity_record"](tampered_path)


# --- ICR-08 and revision identity -------------------------------------------------------------------


def test_selection_narrative_is_conditional():
    text = "\n".join(MARKDOWN)
    assert "each probe reaches its validation minimum at its own step" not in text
    assert "the minimum may lie beyond the window" in text


def test_notebook_metadata_revision_matches_the_code():
    expected = f'NOTEBOOK_REVISION = "{NB["metadata"]["workshop_revision"]}"'
    assert expected in CELLS["Notebook controls"]
    assert NB["metadata"]["workshop_revision"] != "0.1.0-candidate"

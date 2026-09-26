# ruff: noqa: E501,I001
"""Static contract tests for the multi-model image-classification workshop."""
from __future__ import annotations
import ast
import base64
import gzip
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

from vit_classification_pipeline import samples

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_MultiModel_Image_Classification_Workshop.ipynb"

CORE_MODELS = {
    "timm/mobilenetv4_conv_small.e2400_r224_in1k": "331fb803779522b685cf942e15f914fb6741c1eb",
    "timm/resnet50.a1_in1k": "767268603ca0cb0bfe326fa87277f19c419566ef",
    "timm/convnext_tiny.in12k_ft_in1k": "aa096f03029c7f0ec052013f64c819b34f8ad790",
    "timm/vit_base_patch16_224.orig_in21k_ft_in1k": "e0bd370de6799e8d1f47a911174ff4c3708e2323",
    "timm/swinv2_tiny_window8_256.ms_in1k": "650d02aabf05e8adbd060a739ab39e39f53da639",
    "timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k": "81063ecfe9c381a16a19d06f396d6c7011aa426a",
}


def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells():
    return ["".join(cell["source"]) for cell in load()["cells"] if cell["cell_type"] == "code"]


def source():
    return "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])


def as_python(cell):
    return "\n".join(("pass #" + line) if line.lstrip().startswith(("!", "%")) else line for line in cell.splitlines())


def embedded_rows():
    body = source()
    payload = re.search(r'PHOTO_MANIFEST_B64 = "([^"]+)"', body).group(1)
    digest = re.search(r'PHOTO_MANIFEST_SHA256 = "([0-9a-f]{64})"', body).group(1)
    raw = gzip.decompress(base64.b64decode(payload))
    assert hashlib.sha256(raw).hexdigest() == digest
    return json.loads(raw)


def notebook_function(name):
    for cell in code_cells():
        for node in ast.parse(as_python(cell)).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                namespace = {"random": random}
                exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), namespace)
                return namespace[name]
    raise AssertionError(f"{name} not defined in the notebook")


def test_generator_parity():
    subprocess.run([sys.executable, str(REPO / "tools" / "build_multimodel_image_classification_workshop.py"), "--check"], cwd=REPO, check=True)


def test_metadata():
    meta = load()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.1"
    assert meta["notebook_profile"] == "E2E"
    assert meta["notebook_mode"] == "WORKSHOP"
    assert meta["standalone"] is True
    assert meta["default_tier"] == "STANDARD"
    assert meta["worker_required"] is False
    assert meta["credentials_required"] is False
    assert meta["clean_runtime_evidence"] == "pending"


def test_core_models_pinned_at_immutable_revisions():
    body = source()
    for model_id, revision in CORE_MODELS.items():
        assert f'"model_id": "{model_id}"' in body
        assert f'"revision": "{revision}"' in body


def test_code_cells_compile():
    for cell in code_cells():
        compile(as_python(cell), "cell", "exec")


def test_embedded_manifest_matches_the_repository_corpus():
    assert [tuple(row) for row in embedded_rows()] == list(samples.SAMPLE_RECORDS)


def test_builtin_split_matches_the_repository_split():
    rows = embedded_rows()
    split = notebook_function("exact_builtin_split")([{"source_id": row[0], "label": row[1]} for row in rows], seed=samples.SAMPLE_SEED)
    expected = samples.build_sample_dataset([{"id": row[0], "label": row[1]} for row in samples.SAMPLE_RECORDS])
    for name in ("train", "validation", "test"):
        assert [(r["id"], r["source_id"]) for r in split[name]] == [(r["id"], r["source_id"]) for r in expected[name]]
    assert {name: len(split[name]) for name in split} == {"train": 108, "validation": 24, "test": 48}


def test_e2e_and_byod_contract_present():
    body = source()
    for literal in [
        'WORKSHOP_TIER = "STANDARD"',
        "USE_BYOD = False",
        "adapter.safetensors",
        "lowest validation log-loss",
        "fresh reload probability mismatch",
        "frozen_experiment.json",
        "experiment_manifest.json",
        "workshop_summary.json",
        "symlink ZIP member refused",
    ]:
        assert literal in body


def test_no_runtime_repo_dependency():
    body = source()
    for literal in ["git clone ", "pip install -e", "raw.githubusercontent.com/kurtvalcorza", "dimer-backend"]:
        assert literal not in body


def test_committed_notebook_is_clean():
    for cell in load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []

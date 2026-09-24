# ruff: noqa: E501,I001
"""Static contract tests for the multi-model image-classification workshop."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_MultiModel_Image_Classification_Workshop.ipynb"

def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))

def source():
    return "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])

def test_generator_parity():
    subprocess.run([sys.executable, str(REPO / "tools" / "build_multimodel_image_classification_workshop.py"), "--check"], cwd=REPO, check=True)

def test_metadata():
    meta = load()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.1"
    assert meta["notebook_profile"] == "E2E"
    assert meta["notebook_mode"] == "WORKSHOP"
    assert meta["standalone"] is True
    assert meta["clean_runtime_evidence"] == "pending"

def test_six_exact_live_models_present():
    body = source()
    for literal in [
        "timm/resnet50.a1_in1k",
        "timm/mobilenetv4_conv_small.e2400_r224_in1k",
        "timm/convnext_tiny.in12k_ft_in1k",
        "timm/vit_base_patch16_224.orig_in21k_ft_in1k",
        "timm/swinv2_tiny_window8_256.ms_in1k",
        "timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    ]:
        assert literal in body

def test_e2e_and_byod_contract_present():
    body = source()
    for literal in [
        'USE_BYOD = False  # @param',
        'BYOD_ZIP_PATH = ""  # @param',
        "head.safetensors",
        "validation macro_f1",
        "fresh-reload",
        "multimodel_image_classification_provenance.json",
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

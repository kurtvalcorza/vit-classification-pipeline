"""The workshop's model environments are hash-locked uv venvs; nothing is installed into the kernel.

Static and orchestration checks of revision 0.3.0 (2026-10-03). They do not build an environment or run a
model; a hosted run is the evidence that the environments install and run.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
NOTEBOOK = ROOT / "tutorials/DIMER_MultiModel_Image_Classification_Workshop.ipynb"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CODE = {
    "".join(c["source"]).splitlines()[0].removeprefix("# @title "): "".join(c["source"])
    for c in NB["cells"]
    if c["cell_type"] == "code"
}
ENV_CELL = CODE["Create isolated pinned environments"]
CARRIER_CELL = CODE["Carried hash locks of the model environments"]
LOCK_FILES = {
    "common": TOOLS / "image-workshop-common-requirements.lock",
    "swin": TOOLS / "image-workshop-swin-requirements.lock",
}
IN_FILES = {name: path.with_suffix(".in") for name, path in LOCK_FILES.items()}
LINE_LIMIT = 2000


def _load_tool(name):
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(TOOLS))


def _assigned_literal(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in the cell")


def _requirements(text):
    return {
        line.split("==")[0].lower(): line.split("==")[1].split()[0]
        for line in text.splitlines()
        if re.match(r"^[A-Za-z0-9_.-]+==", line)
    }


def test_no_kernel_install_and_no_restart_guard():
    for title, source in CODE.items():
        if title == "Carried hash locks of the model environments":
            # The carried lock headers quote the uv install command as comments; the cell itself runs nothing.
            assert "subprocess" not in source and "import os" not in source
            continue
        assert "pip install" not in source, title
        assert "\"-m\", \"pip\"" not in source and "sys.executable" not in source, title
        # The only install is uv's, into a model environment's own Python, never the kernel's.
        for match in re.finditer(r"\[(\w+), \"pip\", \"install\", \"--python\", (\w+),", source):
            assert match.groups() == ("UV_BINARY", "python"), title
        assert source.count("\"pip\", \"install\"") == len(
            re.findall(r"\[UV_BINARY, \"pip\", \"install\", \"--python\", python,", source)
        ), title
        assert not re.search(r"^\s*[!%]", source, re.M), title
        assert "os.kill" not in source and "Restart session" not in source, title
    markdown = "\n".join("".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "markdown")
    assert "Restart session, then" not in markdown
    assert NB["metadata"]["dimer"]["runtime_environment"]["kernel_installs"] == "none; no restart"


def test_environment_cell_uses_pinned_uv_managed_python_and_hash_locked_wheels():
    assert _assigned_literal(ENV_CELL, "MODEL_ENV_PYTHON") == "3.12.12"
    assert _assigned_literal(ENV_CELL, "UV_VERSION") == "0.12.15"
    url = _assigned_literal(ENV_CELL, "UV_URL")
    assert url.startswith("https://files.pythonhosted.org/") and url.endswith(
        "uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", _assigned_literal(ENV_CELL, "UV_WHEEL_SHA256"))
    for token in ['"--require-hashes"', '"--only-binary", ":all:"', '"--managed-python"', "LOCK_PATHS[name]"]:
        assert token in ENV_CELL, token
    assert 'platform.system() != "Linux" or platform.machine() != "x86_64"' in ENV_CELL
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        assert f'"{name}"' in ENV_CELL
    assert 'CHILD_ENV["MPLBACKEND"] = "Agg"' in ENV_CELL


@pytest.mark.parametrize("name", sorted(LOCK_FILES))
def test_carried_lock_is_the_committed_hash_lock(name):
    namespace = {}
    carried = ast.parse(CARRIER_CELL)
    for node in carried.body:
        if isinstance(node, ast.Assign) and node.targets[0].id in {"CARRIED_LOCKS", "LOCK_SHA256"}:
            namespace[node.targets[0].id] = ast.literal_eval(node.value)
    committed = LOCK_FILES[name].read_bytes()
    assert b"\r" not in committed
    assert namespace["CARRIED_LOCKS"][name].encode("utf-8") == committed
    assert namespace["LOCK_SHA256"][name] == hashlib.sha256(committed).hexdigest()
    assert NB["metadata"]["dimer"]["carried_locks"][name]["sha256"] == hashlib.sha256(committed).hexdigest()


@pytest.mark.parametrize("name", sorted(LOCK_FILES))
def test_every_locked_package_is_hashed_and_direct_pins_match_env_pins(name):
    text = LOCK_FILES[name].read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=[A-Za-z0-9_.-]+==)", text.split("\n", 1)[1])
    packages = [b for b in blocks if re.match(r"^[A-Za-z0-9_.-]+==", b)]
    assert packages
    for block in packages:
        assert "--hash=sha256:" in block, block.splitlines()[0]
    locked = _requirements(text)
    pins = _assigned_literal(ENV_CELL, "ENV_PINS")[name]
    direct = _requirements(IN_FILES[name].read_text(encoding="utf-8"))
    assert direct == {p.split("==")[0].lower(): p.split("==")[1] for p in pins}
    for package, version in direct.items():
        assert locked[package] == version, package


def test_model_stages_run_with_the_environment_python():
    feature = CODE["Run feature extraction one model at a time"]
    assert 'python = ENV_PYTHONS[spec["env"]]' in feature
    assert "str(python), str(feature_runner_path)" in feature
    train = CODE["Train the common probe and verify serialized reload on validation"]
    assert 'common_python=ENV_PYTHONS["common"]' in train
    assert "str(common_python),str(probe_train_path)" in train
    score = CODE["Reload each frozen adapter and score test features"]
    assert "common_python" in score
    for title in (
        "Run feature extraction one model at a time",
        "Train the common probe and verify serialized reload on validation",
        "Reload each frozen adapter and score test features",
    ):
        assert "sys.executable" not in CODE[title], title


def test_ensure_env_builds_from_the_carried_lock(tmp_path):
    source = ENV_CELL.split("ensure_uv()\nneeded_envs")[0]
    namespace = {"WORK_ROOT": tmp_path}
    exec(compile(source, "env cell", "exec"), namespace)
    calls = []

    def fake_run(cmd, label):
        calls.append([str(part) for part in cmd])
        if "venv" in calls[-1]:
            python = tmp_path / "envs" / "common" / "bin" / "python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("")

    lock = tmp_path / "uv" / "requirements-common.lock.txt"
    namespace.update(run_checked=fake_run, LOCK_PATHS={"common": lock}, LOCK_SHA256={"common": "a" * 64})
    python = namespace["ensure_env"]("common")
    uv = str(namespace["UV_BINARY"])
    assert calls[0] == [
        uv, "venv", "--clear", "--managed-python", "--python", "3.12.12", str(tmp_path / "envs/common"),
    ]
    assert calls[1] == [
        uv, "pip", "install", "--python", str(python), "--require-hashes", "--only-binary", ":all:",
        "--index-url", "https://pypi.org/simple", "-r", str(lock),
    ]
    signature = namespace["environment_signature"]("common")
    assert re.fullmatch(r"[0-9a-f]{64}", signature)
    # A rebuilt environment from the same lock is reused; a changed lock rebuilds it.
    calls.clear()
    assert namespace["ensure_env"]("common") == python and calls == []
    namespace["LOCK_SHA256"]["common"] = "b" * 64
    namespace["ensure_env"]("common")
    assert len(calls) == 2 and namespace["environment_signature"]("common") != signature


def test_non_linux_kernels_are_refused_before_any_download(tmp_path, monkeypatch):
    source = ENV_CELL.split("ensure_uv()\nneeded_envs")[0]
    namespace = {"WORK_ROOT": tmp_path}
    exec(compile(source, "env cell", "exec"), namespace)
    monkeypatch.setattr(namespace["platform"], "system", lambda: "Windows")
    monkeypatch.setattr(namespace["urllib"].request, "urlopen", lambda *a, **k: pytest.fail("downloaded"))
    with pytest.raises(RuntimeError, match="Linux x86_64"):
        namespace["ensure_uv"]()


def test_no_notebook_line_exceeds_the_limit():
    for cell in NB["cells"]:
        for line in "".join(cell["source"]).split("\n"):
            assert len(line) <= LINE_LIMIT, (cell["id"], len(line))


def test_split_manifest_literal_equals_the_source_literal():
    generator = _load_tool("build_multimodel_image_classification_workshop")
    title = "# @title Decode the pinned 180-photo manifest"
    original = next(c["source"] for c in generator.CELLS if c["source"].startswith(title))
    expected = _assigned_literal(original, "PHOTO_MANIFEST_B64")
    rendered = CODE["Decode the pinned 180-photo manifest"]
    assert _assigned_literal(rendered, "PHOTO_MANIFEST_B64") == expected
    assert max(len(line) for line in rendered.split("\n")) <= generator.PIECE_LIMIT + 10
    assert len(expected) > generator.PIECE_LIMIT


def test_split_long_literals_round_trips_and_leaves_short_lines_alone():
    generator = _load_tool("build_multimodel_image_classification_workshop")
    value = "x" * 2500
    source = f'a = 1\nBIG = "{value}"\nb = "short"'
    rendered = generator.split_long_literals(source)
    namespace = {}
    exec(compile(rendered, "split", "exec"), namespace)
    assert namespace["BIG"] == value and namespace["b"] == "short"
    assert max(len(line) for line in rendered.split("\n")) <= generator.PIECE_LIMIT + 10


def test_revision_log_keeps_every_review_fix():
    revisions = NB["metadata"]["dimer"]["review_revisions"]
    latest = revisions[-1]
    assert latest["revision"] == NB["metadata"]["workshop_revision"] == "0.3.0-candidate"
    assert latest["fixes_kept"] == [f"ICR-0{i}" for i in range(1, 9)]
    assert f'NOTEBOOK_REVISION = "{latest["revision"]}"' in CODE["Notebook controls"]
    assert "Linux x86_64 only" in NB["metadata"]["dimer"]["runtime_environment"]["platform"]

"""Row 108 review fixes (2026-10-17): the pre-registered grid, the fresh test and the score-once wiring.

SCORE_ONCE_RULE items 1-8: `preregistered` over ALL fixed settings, exploratory runs never touch the fresh test,
an identical re-run in the same runtime is a reproduction that never overwrites, outputs carry `test_look` and
`settings_differing`. Section 8 is executed with stand-ins (no model, no download).
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vit_classification_pipeline import protocol
from vit_classification_pipeline.protocol import (
    GRID,
    PREREGISTERED_SETTINGS,
    LookLedger,
    config_record,
    grid_check,
    paired_bootstrap,
    preregistration_differences,
    reading_verdict,
    scoring_mode,
    select_config,
)
from vit_classification_pipeline.samples import (
    FRESH_TEST_DIGEST,
    FRESH_TEST_RECORDS,
    SAMPLE_RECORDS,
    check_fresh_disjoint,
    fresh_test_digest,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "vit_classification_colab.ipynb"
STEM = "vit_classification"
SPLITS = {"train": "d-train", "validation": "d-val"}


def _adaptation(blocks, lr, *, probe=0.162, best=None, val_acc=1.0, epochs=4):
    history = [{"epoch": 0, "val": {"log_loss": probe, "accuracy": val_acc}}]
    for e in range(1, epochs + 1):
        history.append({"epoch": e, "val": {"log_loss": probe + 0.003 * e, "accuracy": val_acc}})
    best_epoch = 0
    if best is not None:
        history[2]["val"]["log_loss"] = best
        best_epoch = 2
    return {
        "history": history, "best_epoch": best_epoch, "policy": "frozen backbone + linear probe" if best_epoch == 0
        else f"unfrozen last {blocks} blocks + linear head", "trainable_blocks": blocks, "lr": lr, "epochs": epochs,
        "batch_size": 8, "probe_steps": 300, "probe_lr": 0.01, "seed": 0, "min_improvement": 0.02,
        "probe_val_log_loss": probe,
    }


def _grid(**best):
    rows = {}
    for config in GRID:
        result = _adaptation(config["trainable_blocks"], config["lr"], best=best.get(config["name"]))
        rows[config["name"]] = config_record(config, result, split_digests=SPLITS, fingerprint="fp-" + config["name"])
    return rows


# ---- the fresh test pins (R108-D1)
def test_fresh_test_is_pinned_disjoint_and_sixteen_per_species():
    out = check_fresh_disjoint()
    assert out["n"] == 96 and out["digest_pinned"] and not any(out["shared_with_corpus"].values())
    assert fresh_test_digest(FRESH_TEST_RECORDS) == FRESH_TEST_DIGEST
    per = {}
    for r in FRESH_TEST_RECORDS:
        per[r[1]] = per.get(r[1], 0) + 1
    assert set(per.values()) == {16} and len(per) == 6
    by_species = {}
    for r in FRESH_TEST_RECORDS:
        by_species.setdefault(r[1], []).append(r[2])
    assert all(ids == sorted(ids) for ids in by_species.values())  # ascending photo id
    draw = json.loads((ROOT / "docs" / "fresh-test-draw.json").read_text(encoding="utf-8"))
    assert draw["selection_digest"] == FRESH_TEST_DIGEST and draw["n_records"] == 96
    assert [r["photo_id"] for r in draw["records"]] == [r[2] for r in FRESH_TEST_RECORDS]


def test_a_changed_fresh_table_is_refused():
    changed = list(FRESH_TEST_RECORDS)
    changed[0] = (*changed[0][:5], changed[0][5] + 1, *changed[0][6:])
    with pytest.raises(ValueError, match="not the pinned"):
        check_fresh_disjoint(changed)
    with pytest.raises(ValueError, match="not the pinned"):
        check_fresh_disjoint([*FRESH_TEST_RECORDS[:-1], SAMPLE_RECORDS[0]])


# ---- rule items 1 and 7: preregistered over ALL fixed settings
@pytest.mark.parametrize(
    ("key", "value"),
    [("epochs", 5), ("batch_size", 4), ("adapt_seed", 1), ("probe_lr", 0.02), ("probe_steps", 100),
     ("data_source", "byod"), ("split_seed", 7), ("split_sizes", {"train": 17, "validation": 4, "test": 8}),
     ("min_improvement", 0.0), ("run_grid", False), ("experiment", True), ("corpus_pins", "x"),
     ("fresh_test_digest", "y"), ("grid", [dict(GRID[0])])],
)
def test_any_changed_setting_is_listed_and_makes_the_run_exploratory(key, value):
    current = {**PREREGISTERED_SETTINGS, key: value}
    differing = preregistration_differences(current)
    assert differing == [key]
    mode = scoring_mode(differences=differing, grid={"complete": True}, state_matches=True, masked=False)
    assert mode == "exploratory"


def test_a_missing_setting_differs_and_the_registered_one_does_not():
    assert preregistration_differences(dict(PREREGISTERED_SETTINGS)) == []
    assert "epochs" in preregistration_differences({k: v for k, v in PREREGISTERED_SETTINGS.items() if k != "epochs"})


# ---- the grid (R108-D2)
def test_selection_keeps_c0_unless_a_challenger_wins_by_the_margin():
    assert select_config(_grid())["selected"] == "C0"
    assert select_config(_grid(C1=0.150))["selected"] == "C0"  # 0.012 better than C0's probe: below the margin
    assert select_config(_grid(C1=0.140))["selected"] == "C1"
    assert select_config(_grid(C1=0.140, C2=0.130))["selected"] == "C2"
    assert select_config(_grid(C1=0.130, C2=0.130))["selected"] == "C2"  # tie: the smaller trainable scope


def test_a_challenger_with_lower_validation_accuracy_does_not_replace_c0():
    rows = _grid(C1=0.100)
    rows["C1"]["val_accuracy"] = 0.95
    assert select_config(rows)["selected"] == "C0"


def test_a_missing_or_failed_config_makes_the_grid_incomplete():
    rows = _grid()
    assert grid_check(rows, PREREGISTERED_SETTINGS, SPLITS)["complete"]
    del rows["C2"]
    status = grid_check(rows, PREREGISTERED_SETTINGS, SPLITS)
    assert not status["complete"] and "C2: not run" in status["problems"]
    rows["C2"] = config_record(GRID[2], None, split_digests=SPLITS, error="out of memory: CUDA")
    assert not grid_check(rows, PREREGISTERED_SETTINGS, SPLITS)["complete"]
    assert scoring_mode(differences=[], grid=grid_check(rows, PREREGISTERED_SETTINGS, SPLITS), state_matches=True,
                        masked=False) == "exploratory"


def test_a_grid_run_on_another_split_or_with_other_settings_does_not_match():
    rows = _grid()
    assert not grid_check(rows, PREREGISTERED_SETTINGS, {"train": "other", "validation": "d-val"})["complete"]
    assert not grid_check(rows, {**PREREGISTERED_SETTINGS, "epochs": 5}, SPLITS)["complete"]


# ---- the reading rule, fixed before any fresh score
def test_paired_bootstrap_and_verdict():
    rows = [{"id": str(i), "correct": i % 2 == 0, "nll": 0.5} for i in range(96)]
    same = paired_bootstrap(rows, rows)
    assert same["accuracy_ci95"] == [0.0, 0.0] and same["n_boot"] == protocol.BOOTSTRAP_N
    assert reading_verdict(same, same_model=True) == "same model (probe kept)"
    better = [{**r, "correct": True, "nll": 0.1} for r in rows]
    boot = paired_bootstrap(better, rows)
    assert boot["accuracy_ci95"][0] > 0 and reading_verdict(boot, same_model=False) == "unfreeze better"
    assert reading_verdict(paired_bootstrap(rows, better), same_model=False) == "unfreeze worse"
    assert paired_bootstrap(better, rows) == boot  # fixed seed: reproducible


# ---- rule items 3 and 5: reproductions and exploratory looks never overwrite
def test_look_ledger_labels_a_repeat_as_a_reproduction(tmp_path):
    ledger = LookLedger(tmp_path / "ledger.json")
    assert ledger.next_look("preregistered") == (protocol.TEST_LOOK_PREREGISTERED, "")
    assert ledger.next_look("preregistered") == (protocol.TEST_LOOK_REPRODUCTION, "_reproduction_1")
    assert LookLedger(tmp_path / "ledger.json").next_look("preregistered")[1] == "_reproduction_2"  # survives re-creation
    assert ledger.next_look("exploratory") == (protocol.TEST_LOOK_EXPLORATORY, "_exploratory_1")
    assert ledger.next_look("masked")[1] == "_masked_1"


def test_recorded_evaluation_is_none_until_the_hosted_run():
    assert protocol.RECORDED_EVALUATION is None


# ---- Section 8 executed with stand-ins
def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _cell(marker):
    hits = [c for c in _cells() if marker in c]
    assert len(hits) == 1, marker
    return hits[0]


class _Pipe:
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.evaluated = []

    def state_fingerprint(self):
        return self.fingerprint

    def evaluate(self, records, per_record=False):
        self.evaluated.append(len(records))
        out = {"accuracy": 0.75, "macro_f1": 0.7, "log_loss": 0.6, "n": len(records),
               "per_class": {"a": {"recall": 1.0}, "b": {"recall": 0.5}}, "confusion": [[1, 0], [1, 1]]}
        if per_record:
            out["per_record"] = [{"id": r["id"], "label": r["label"], "predicted": r["label"], "correct": True, "nll": 0.2}
                                 for r in records]
        return out


def _namespace(tmp_path, monkeypatch, *, split_seed=42, grid=None, masked=False):
    import vit_classification_pipeline as pkg
    from vit_classification_pipeline import samples

    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    if masked:
        monkeypatch.setenv("DIMER_MASK_FRESH_TEST", "1")
    else:
        monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    fetched = []

    def fetch_fresh_test(**kwargs):
        fetched.append(kwargs)
        return [{"id": f"f{i}", "label": "a", "image": i} for i in range(96)]

    ns = {name: getattr(protocol, name) for name in dir(protocol) if not name.startswith("__")}
    ns.update({name: getattr(samples, name) for name in dir(samples) if not name.startswith("__")})
    ns.update({name: getattr(pkg, name) for name in pkg.__all__})
    rows = grid if grid is not None else _grid()
    ns.update({
        "os": __import__("os"), "json": json, "Path": Path,
        "RUN_VALID": True, "RunStopped": RuntimeError, "LOOK_LEDGER": LookLedger(tmp_path / "work" / "ledger.json"),
        "USE_BYOD": False, "SPLIT_SEED": split_seed, "PROBE_STEPS": 300, "PROBE_LR": 0.01, "EPOCHS": 4, "BATCH_SIZE": 8,
        "RUN_GRID": True, "ADAPT_SEED": 0, "grid_to_run": list(GRID),
        "dataset_manifests": {"train": {"digest": "d-train", "label_counts": {"a": 18, "b": 18}},
                              "validation": {"digest": "d-val", "label_counts": {"a": 4, "b": 4}},
                              "test": {"digest": "d-test", "label_counts": {"a": 8, "b": 8}}},
        "grid_rows": rows, "selection": select_config(rows), "pipe": _Pipe("fp-C0"),
        "test_records": [{"id": "t1", "label": "a"}], "val_records": [{"id": "v1", "label": "a"}],
        "floor": {"accuracy": 0.5, "macro_f1": 0.3}, "baseline_knn": {"accuracy": 0.6, "macro_f1": 0.5,
                                                                      "per_class": {"a": {"recall": 1.0}, "b": {"recall": 0.0}}},
        "frozen_test": {"accuracy": 0.75, "macro_f1": 0.7, "log_loss": 0.6, "per_class": {"a": {"recall": 1.0}, "b": {"recall": 0.5}}},
        "classes": ["a", "b"], "adapt_result": {**_adaptation(2, 3e-5), "n_total": 1}, "frozen_verdict": "above floor",
        "splits": {"train": [{"image": "x"}]}, "image_digest": lambda image: f"px-{image}",
        "fetch_fresh_test": fetch_fresh_test, "probe_artifact_dir": tmp_path / "probe", "probe_fingerprint": "fp-probe",
        "WEIGHTS_DIR": tmp_path, "brief": lambda m: {"accuracy": m["accuracy"], "n": m["n"]}, "data_source": "sample",
        "disjoint": {}, "overlap": {}, "probe_result": {"history": []}, "adapt_seconds": 1.0,
    })
    settings_cell = _cell("def current_settings():")
    start = settings_cell.index("def current_settings():")
    end = settings_cell.index("\n\ngrid_to_run", start)
    exec(compile(settings_cell[start:end], "current_settings", "exec"), ns)
    return ns, fetched


SECTION_8 = "test_look, look_suffix = LOOK_LEDGER.next_look(mode)"


def _report(tmp_path, suffix=""):
    return json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report{suffix}.json").read_text(encoding="utf-8"))


def test_section_8_scores_the_fresh_test_once_and_a_rerun_is_a_reproduction(tmp_path, monkeypatch):
    ns, fetched = _namespace(tmp_path, monkeypatch)
    exec(_cell(SECTION_8), ns)
    first = _report(tmp_path)
    assert first["preregistered"] and first["settings_differing"] == []
    assert first["test_look"] == protocol.TEST_LOOK_PREREGISTERED and first["fresh_test"]["scored"]
    assert first["fresh_test"]["verdict"] == "same model (probe kept)" and len(fetched) == 1
    assert first["exposed_development_test"]["comparison"]["verdicts"] == {
        "frozen_vs_floor": "above floor", "selected_vs_floor": "above floor", "selected_vs_frozen_accuracy": "no gain"}
    exec(_cell(SECTION_8), ns)
    again = _report(tmp_path, "_reproduction_1")
    assert again["test_look"] == protocol.TEST_LOOK_REPRODUCTION
    assert _report(tmp_path) == first  # the first file is never overwritten


def test_section_8_with_a_changed_split_seed_never_touches_the_fresh_test(tmp_path, monkeypatch):
    ns, fetched = _namespace(tmp_path, monkeypatch, split_seed=7)
    exec(_cell(SECTION_8), ns)
    report = _report(tmp_path, "_exploratory_1")
    assert not report["preregistered"] and report["settings_differing"] == ["split_seed"]
    assert report["test_look"] == protocol.TEST_LOOK_EXPLORATORY and not report["fresh_test"]["scored"]
    assert fetched == [] and not (tmp_path / "outputs" / f"{STEM}_evaluation_report.json").exists()


def test_section_8_with_a_missing_grid_config_never_touches_the_fresh_test(tmp_path, monkeypatch):
    rows = _grid()
    del rows["C1"]
    ns, fetched = _namespace(tmp_path, monkeypatch, grid=rows)
    exec(_cell(SECTION_8), ns)
    report = _report(tmp_path, "_exploratory_1")
    assert not report["preregistered"] and "C1: not run" in report["grid"]["status"]["problems"]
    assert fetched == []


def test_section_8_with_a_changed_epoch_count_or_model_state_is_exploratory(tmp_path, monkeypatch):
    ns, fetched = _namespace(tmp_path, monkeypatch)
    ns["EPOCHS"] = 5
    exec(_cell(SECTION_8), ns)
    assert _report(tmp_path, "_exploratory_1")["settings_differing"] == ["epochs"]
    ns["EPOCHS"] = 4
    ns["pipe"] = _Pipe("fp-something-else")
    exec(_cell(SECTION_8), ns)
    report = _report(tmp_path, "_exploratory_2")
    assert report["settings_differing"] == [] and not report["fresh_test"]["scored"] and fetched == []


def test_section_8_masked_for_a_development_check_never_fetches(tmp_path, monkeypatch):
    ns, fetched = _namespace(tmp_path, monkeypatch, masked=True)
    exec(_cell(SECTION_8), ns)
    report = _report(tmp_path, "_masked_1")
    assert report["preregistered"] and report["test_look"] == protocol.TEST_LOOK_MASKED and fetched == []


def test_section_10_makes_later_section_8_runs_exploratory(tmp_path, monkeypatch):
    ns, fetched = _namespace(tmp_path, monkeypatch)
    ns["experiment_ran"] = True
    exec(_cell(SECTION_8), ns)
    assert _report(tmp_path, "_exploratory_1")["settings_differing"] == ["experiment"] and fetched == []


# ---- R108-M1/M2/m1/m2/m3 prose
def test_false_claims_are_gone_and_the_disclosures_are_present():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    for phrase in ("never used for training or policy selection", "never by the test split",
                   "label space the checkpoint does not have", "names three of the six", "Kaggle T4 release run",
                   "independent test split", "independent split", "here it was"):
        assert phrase not in text, phrase
    for phrase in ("exposed development test", "ImageNet-21k", "n01536035", "European goldfinch", "72.9 %", "66.7 %",
                   "docs/vit-preregistration.md", "settings_differing"):
        assert phrase in text, phrase
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.5" and nb["metadata"]["dimer"]["notebook_kind"] == "TUTORIAL"


def test_the_preregistration_was_committed_before_the_code():
    text = (ROOT / "docs" / "vit-preregistration.md").read_text(encoding="utf-8")
    assert FRESH_TEST_DIGEST in text and "72.9 %" in text and "n_boot" in text
    assert re.search(r"C1 \| 4 \| 1e-4", text) and re.search(r"C2 \| 2 \| 3e-4", text)

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


# ---- the reading rule (amendment 1): one primary endpoint, observer-cluster bootstrap
def test_cluster_bootstrap_primary_endpoint_and_verdict():
    rows = [{"id": str(i), "correct": i % 2 == 0, "nll": 0.5} for i in range(96)]
    observers = [f"o{i % 79}" for i in range(96)]
    same = paired_bootstrap(rows, rows, clusters=observers)
    assert same["primary_accuracy_ci95"] == [0.0, 0.0] and same["n_clusters"] == 79
    assert same["resampling_unit"] == "observer" and same["n_boot"] == protocol.BOOTSTRAP_N
    assert reading_verdict(same, same_model=True) == "same model"
    better = [{**r, "correct": True, "nll": 0.1} for r in rows]
    boot = paired_bootstrap(better, rows, clusters=observers)
    assert boot["primary_accuracy_ci95"][0] > 0 and reading_verdict(boot, same_model=False) == "improved"
    assert reading_verdict(paired_bootstrap(rows, better, clusters=observers), same_model=False) == "worse"
    assert paired_bootstrap(better, rows, clusters=observers) == boot  # fixed seed: reproducible
    # log-loss is secondary: a log-loss gain alone never makes a verdict
    lower_nll = [{**r, "nll": 0.1} for r in rows]
    only_nll = paired_bootstrap(lower_nll, rows, clusters=observers)
    assert only_nll["secondary_log_loss_ci95"][1] < 0
    assert reading_verdict(only_nll, same_model=False) == "no detectable difference"
    three_down = [{**r, "correct": False} if i in (0, 2, 4) else r for i, r in enumerate(rows)]
    assert reading_verdict(paired_bootstrap(three_down, rows, clusters=observers), same_model=False) == "no detectable difference"


def test_amendment_1_was_committed_alone_and_states_the_rule():
    text = (ROOT / "docs" / "vit-preregistration-amendment-1.md").read_text(encoding="utf-8")
    assert "resampled by observer" in text and "Primary endpoint" in text and "20261017" in text
    assert "No fresh-test score exists when this is written" in text and "C1" in text


# ---- look counting (F2): only completed, written scorings count; outputs decide the slot too
def test_ledger_plans_without_recording_and_records_after_the_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger = LookLedger(tmp_path / "work" / "ledger.json")
    assert ledger.plan("preregistered") == (protocol.TEST_LOOK_PREREGISTERED, "")
    assert ledger.plan("preregistered") == (protocol.TEST_LOOK_PREREGISTERED, "")  # planning spends nothing
    protocol.atomic_write_text(tmp_path / "outputs" / f"{STEM}_evaluation_report.json", "{}")
    ledger.record("preregistered", "")
    assert ledger.plan("preregistered") == (protocol.TEST_LOOK_REPRODUCTION, "_reproduction_1")
    assert ledger.plan("exploratory") == (protocol.TEST_LOOK_EXPLORATORY, "_exploratory_1")
    assert ledger.plan("masked")[1] == "_masked_1"


def test_e1_an_empty_or_corrupt_ledger_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "ledger.json").write_text("", encoding="utf-8")
    ledger = LookLedger(tmp_path / "work" / "ledger.json")
    assert ledger.plan("preregistered")[1] == "" and ledger.touched() == 1
    (tmp_path / "work" / "ledger.json").write_text("{not json", encoding="utf-8")
    assert ledger.plan("exploratory")[1] == "_exploratory_1"


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
    device = "cpu"

    def __init__(self, fingerprint, accuracy=0.75):
        self.fingerprint = fingerprint
        self.accuracy = accuracy
        self.evaluated = []

    def knn_baseline(self, train, test, k=5):
        return {"accuracy": 0.7, "macro_f1": 0.65}

    @classmethod
    def from_artifact(cls, *a, **k):
        return cls("fp-probe")

    def state_fingerprint(self):
        return self.fingerprint

    def evaluate(self, records, per_record=False):
        self.evaluated.append(len(records))
        out = {"accuracy": self.accuracy, "macro_f1": 0.7, "log_loss": 0.6, "n": len(records),
               "per_class": {"a": {"recall": 1.0}, "b": {"recall": 0.5}}, "confusion": [[1, 0], [1, 1]]}
        if per_record:
            out["per_record"] = [{"id": r["id"], "label": r["label"], "predicted": r["label"], "correct": True, "nll": 0.2}
                                 for r in records]
        return out


def _namespace(tmp_path, monkeypatch, *, split_seed=42, grid=None, masked=False, fail_fetch=False):
    import vit_classification_pipeline as pkg
    from vit_classification_pipeline import samples

    (tmp_path / "outputs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    if masked:
        monkeypatch.setenv("DIMER_MASK_FRESH_TEST", "1")
    else:
        monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    fetched = []

    def fetch_fresh_test(**kwargs):
        fetched.append(kwargs)
        if fail_fetch:
            raise OSError("network unreachable")
        return [{"id": f"f{i}", "label": "a" if i % 2 else "b", "image": i, "observer": f"o{i % 79}"} for i in range(96)]

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
        "train_records": [{"id": "r1", "label": "a"}, {"id": "r2", "label": "b"}],
        "ViTClassificationPipeline": _Pipe, "numeric_settings": lambda device: {"device": device, "stand_in": True},
        "math": __import__("math"),
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
    digests = {id(ns["train_records"]): "d-train", id(ns["val_records"]): "d-val"}
    ns["dataset_digest"] = lambda records: digests.get(id(records), "edited")
    return ns, fetched


SECTION_8 = "test_look, look_suffix = LOOK_LEDGER.plan(mode)"


def _report(tmp_path, suffix=""):
    return json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report{suffix}.json").read_text(encoding="utf-8"))


def test_section_8_scores_the_fresh_test_once_and_a_rerun_is_a_reproduction(tmp_path, monkeypatch):
    ns, fetched = _namespace(tmp_path, monkeypatch)
    exec(_cell(SECTION_8), ns)
    first = _report(tmp_path)
    assert first["preregistered"] and first["settings_differing"] == []
    assert first["test_look"] == protocol.TEST_LOOK_PREREGISTERED and first["fresh_test"]["scored"]
    assert first["fresh_test"]["verdict"] == "same model" and len(fetched) == 1
    assert first["fresh_test"]["paired_bootstrap"]["resampling_unit"] == "observer"
    assert first["fresh_test"]["references_decide_nothing"]["knn5_frozen_base"]["accuracy"] == 0.7
    assert first["exposed_development_test"]["comparison"]["descriptives_decide_nothing"] == {
        "frozen_vs_floor": "above floor", "selected_vs_floor": "above floor", "selected_minus_frozen_photographs": 0}
    assert first["preregistered"] and not first["masked"] and first["fresh_test_scored"] and first["matches_first"] is None
    exec(_cell(SECTION_8), ns)
    again = _report(tmp_path, "_reproduction_1")
    assert again["test_look"] == protocol.TEST_LOOK_REPRODUCTION
    assert again["matches_first"] == {"first_report_readable": True, "selected_config": True, "fingerprint": True, "per_record_identical": True}
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
    # F3: a masked development check is not a pre-registered scoring
    assert not report["preregistered"] and report["preregistered_settings_hold"] and report["masked"]
    assert report["test_look"] == protocol.TEST_LOOK_MASKED and not report["fresh_test_scored"] and fetched == []


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


# ---- the reviewer's probe cases E2-E4 (F2) and the library mask (F9)
def test_e2_losing_work_after_the_first_scoring_never_overwrites_the_main_report(tmp_path, monkeypatch):
    import shutil

    ns, fetched = _namespace(tmp_path, monkeypatch)
    exec(_cell(SECTION_8), ns)
    first = _report(tmp_path)
    shutil.rmtree(tmp_path / "work")
    ns["pipe"].accuracy = 0.5
    exec(_cell(SECTION_8), ns)
    assert _report(tmp_path) == first
    assert _report(tmp_path, "_reproduction_1")["test_look"] == protocol.TEST_LOOK_REPRODUCTION


def test_e3_a_truncated_main_report_is_rewritten_unless_the_ledger_proves_a_look(tmp_path, monkeypatch):
    ns, fetched = _namespace(tmp_path, monkeypatch)
    main = tmp_path / "outputs" / f"{STEM}_evaluation_report.json"
    main.write_text('{"test_look": "pre-reg', encoding="utf-8")  # an interrupted non-atomic write
    exec(_cell(SECTION_8), ns)
    assert _report(tmp_path)["test_look"] == protocol.TEST_LOOK_PREREGISTERED
    main.write_text("", encoding="utf-8")  # unreadable again, but the ledger now proves a completed look
    exec(_cell(SECTION_8), ns)
    assert _report(tmp_path, "_reproduction_1")["test_look"] == protocol.TEST_LOOK_REPRODUCTION


def test_e4_a_failed_fetch_is_not_a_look_and_the_next_scoring_is_the_main_report(tmp_path, monkeypatch):
    ns, fetched = _namespace(tmp_path, monkeypatch, fail_fetch=True)
    with pytest.raises(OSError):
        exec(_cell(SECTION_8), ns)
    assert not list((tmp_path / "outputs").glob("*evaluation_report*"))
    assert ns["LOOK_LEDGER"].touches() == 1
    ns2, fetched2 = _namespace(tmp_path / "again", monkeypatch)
    ns2["LOOK_LEDGER"] = ns["LOOK_LEDGER"].__class__(tmp_path / "work" / "ledger.json", tmp_path / "again" / "outputs")
    exec(_cell(SECTION_8), ns2)
    report = _report(tmp_path / "again")
    assert report["test_look"] == protocol.TEST_LOOK_PREREGISTERED and report["fresh_test_fetches_in_this_runtime"] == 2


def test_fetch_fresh_test_itself_refuses_while_masked(monkeypatch):
    from vit_classification_pipeline import samples

    monkeypatch.setenv("DIMER_MASK_FRESH_TEST", "1")
    with pytest.raises(RuntimeError, match="masked"):
        samples.fetch_fresh_test(fetcher=lambda url: pytest.fail("fetched while masked"))


def test_edited_training_records_make_the_run_exploratory(tmp_path, monkeypatch):
    ns, fetched = _namespace(tmp_path, monkeypatch)
    ns["train_records"] = ns["train_records"][:1]  # F15: edited after Section 4
    exec(_cell(SECTION_8), ns)
    report = _report(tmp_path, "_exploratory_1")
    assert "trained_records" in report["settings_differing"] and fetched == []


def test_f14_a_byod_set_at_the_floor_is_exploratory_and_still_writes_its_report(tmp_path, monkeypatch):
    ns, fetched = _namespace(tmp_path, monkeypatch)
    ns["USE_BYOD"] = True
    ns["floor"] = {"accuracy": 0.8, "macro_f1": 0.7}
    ns["frozen_verdict"] = "at or below floor"
    ns["pipe"].accuracy = 0.5
    exec(_cell(SECTION_8), ns)
    report = _report(tmp_path, "_exploratory_1")
    assert report["settings_differing"] == ["data_source"] and fetched == []
    assert report["exposed_development_test"]["comparison"]["descriptives_decide_nothing"] == {
        "frozen_vs_floor": "at or below floor", "selected_vs_floor": "at or below floor", "selected_minus_frozen_photographs": -0}

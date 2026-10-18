"""The tutorial's pre-registered selection and fresh-test rules (docs/vit-preregistration.md, 2026-10-18).

`PREREGISTERED_SETTINGS` lists every setting the pre-registration fixes. A run is `preregistered` only when all
of them hold, the validation grid is complete and matches them, and the model in hand is the selected system;
only such a run scores the fresh test. Every other run is exploratory: validation and the labelled exposed
development test only, written to `*_exploratory_<n>` files. The evaluation of record is the 2026-10-18 hosted
Colab T4 run (`RECORDED_EVALUATION`; docs/release-verification.md), so every later pre-registered scoring, in this
runtime or a new one, is a reproduction: it writes `*_reproduction_<n>` files, never the main ones, and is compared
with the record (`compare_with_record`).
"""

# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .samples import FRESH_TEST_DIGEST, SAMPLE_RECORDS, SAMPLE_SPLIT

MIN_IMPROVEMENT = 0.02  # log-loss margin for a challenger config and for an unfreeze over the probe
GRID: tuple[dict[str, Any], ...] = (
    {"name": "C0", "trainable_blocks": 2, "lr": 3e-5},
    {"name": "C1", "trainable_blocks": 4, "lr": 1e-4},
    {"name": "C2", "trainable_blocks": 2, "lr": 3e-4},
)
DEFAULT_CONFIG = "C0"
TIE_ORDER = ("C0", "C2", "C1")  # ties go to the default, then to the smaller trainable scope
TIE_TOLERANCE = 1e-9
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261017


def corpus_pins_digest(records: Sequence[Sequence[Any]] = SAMPLE_RECORDS) -> str:
    """sha256 over the pinned corpus table (id, label, photo id, bytes, sha256 per row)."""
    text = "\n".join(f"{r[0]}:{r[1]}:{r[2]}:{r[5]}:{r[6]}" for r in records)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data_source": "sample",
    "corpus_pins": corpus_pins_digest(),
    "split_seed": 42,
    "split_sizes": dict(SAMPLE_SPLIT),
    "probe_steps": 300,
    "probe_lr": 0.01,
    "epochs": 4,
    "batch_size": 8,
    "adapt_seed": 0,
    "min_improvement": MIN_IMPROVEMENT,
    "grid": [dict(c) for c in GRID],
    "run_grid": True,
    "experiment": False,
    "fresh_test_digest": FRESH_TEST_DIGEST,
}

# The hosted T4 run that is the evaluation of record (REL13 label, identity, numbers), re-derived from the executed
# notebook in docs/execution-evidence/2026-10-18-1231036/ (tests/test_score_once.py checks every number against it).
# With it set, no runtime writes the main report again: every pre-registered scoring is a reproduction.
RECORDED_EVALUATION: dict[str, Any] | None = {
    "label": "pre-registered evaluation of record",
    "date": "2026-10-18",
    "runner": "Colab CLI 0.7.4, fresh hosted Colab Tesla T4 runtime (cuda:0, float32, TF32 off)",
    "commit": "12310362e09bcf47dda3000266a52c072c91f891",
    "notebook_blob": "94e67934a115a3a3100c160681f7493753145a6d",
    "evidence": "docs/execution-evidence/2026-10-18-1231036/",
    "preregistered": True,
    "settings_differing": [],
    "selected_config": "C1",
    "selected_policy": "unfrozen last 4 blocks + linear head",
    "best_epoch": 4,
    "grid_val_log_loss": {"C0": 0.1621, "C1": 0.1242, "C2": 0.1375},
    "grid_val_accuracy": {"C0": 0.9167, "C1": 0.9583, "C2": 0.9583},
    "fresh_test": {
        "n": 96,
        "selected": {"accuracy": 0.8125, "correct": 78, "macro_f1": 0.8115, "log_loss": 0.52},
        "frozen_probe": {"accuracy": 0.8333, "correct": 80, "macro_f1": 0.8315, "log_loss": 0.4922},
        "primary_accuracy_difference": -0.0208,
        "primary_accuracy_ci95": [-0.0632, 0.0204],
        "n_clusters": 79,
        "discordant_images": {"selected_only": 1, "probe_only": 3},
        "secondary_log_loss_difference": 0.0277,
        "secondary_log_loss_ci95": [-0.044, 0.1159],
        "verdict": "no detectable difference",
        "references_decide_nothing": {"majority_floor": {"accuracy": 0.1667, "macro_f1": 0.0476},
                                      "knn5_frozen_base": {"accuracy": 0.8542, "correct": 82, "macro_f1": 0.854}},
    },
    "exposed_development_test": {"n": 48, "majority_floor": 0.1667, "knn5": 0.75, "frozen_policy": 0.7917, "selected": 0.7292},
    "reload_parity_accuracy": 0.729167,
}

TEST_LOOK_PREREGISTERED = "pre-registered evaluation (this runtime)"
TEST_LOOK_REPRODUCTION = "reproduction"
TEST_LOOK_EXPLORATORY = "not scored (exploratory)"
TEST_LOOK_MASKED = "not scored (fresh test masked for a development check)"
UNIFORM_LOG_LOSS_6 = 1.7918  # ln 6: the log-loss of a uniform guess over the six species (a reference point)


def preregistration_differences(current: Mapping[str, Any]) -> list[str]:
    """Every pre-registered setting whose current value differs (a missing setting differs)."""
    missing = object()
    return [key for key, value in PREREGISTERED_SETTINGS.items() if current.get(key, missing) != value]


def config_record(
    config: Mapping[str, Any],
    adaptation: Mapping[str, Any] | None,
    *,
    split_digests: Mapping[str, str],
    fingerprint: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """One grid row, validation only: the settings it ran with and its selected system's validation scores."""
    row: dict[str, Any] = {"name": config["name"], "config": dict(config), "split_digests": dict(split_digests)}
    if error is not None or adaptation is None:
        row.update({"status": "failed", "error": error or "no result"})
        return row
    history = adaptation["history"]
    chosen = next(e for e in history if e["epoch"] == adaptation["best_epoch"])
    row.update(
        {
            "status": "complete",
            "settings": {
                k: adaptation[k]
                for k in ("trainable_blocks", "lr", "epochs", "batch_size", "probe_steps", "probe_lr", "seed",
                          "min_improvement")
            },
            "policy": adaptation["policy"],
            "best_epoch": adaptation["best_epoch"],
            "val_log_loss": chosen["val"]["log_loss"],
            "val_accuracy": chosen["val"]["accuracy"],
            "probe_val_log_loss": adaptation.get("probe_val_log_loss"),
            "val_log_loss_by_epoch": [e["val"]["log_loss"] for e in history if e.get("val")],
            "fingerprint": fingerprint,
        }
    )
    return row


def grid_check(
    grid: Mapping[str, Mapping[str, Any]] | None,
    settings: Mapping[str, Any],
    split_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Is the grid complete, run with the current fixed settings, on the current split?"""
    problems: list[str] = []
    grid = grid or {}
    for config in GRID:
        row = grid.get(config["name"])
        if row is None:
            problems.append(f"{config['name']}: not run")
            continue
        if row.get("status") != "complete":
            problems.append(f"{config['name']}: {row.get('status')} ({row.get('error')})")
            continue
        expected = {
            "trainable_blocks": config["trainable_blocks"],
            "lr": config["lr"],
            "epochs": settings.get("epochs"),
            "batch_size": settings.get("batch_size"),
            "probe_steps": settings.get("probe_steps"),
            "probe_lr": settings.get("probe_lr"),
            "seed": settings.get("adapt_seed"),
            "min_improvement": settings.get("min_improvement"),
        }
        if row["settings"] != expected:
            problems.append(f"{config['name']}: ran with {row['settings']}, current settings imply {expected}")
        if dict(row["split_digests"]) != dict(split_digests):
            problems.append(f"{config['name']}: ran on a different split")
    extra = sorted(set(grid) - {c["name"] for c in GRID})
    if extra:
        problems.append(f"configs outside the pre-registered grid: {extra}")
    return {"complete": not problems, "problems": problems}


def select_config(grid: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The pre-registered choice among completed configs (validation only)."""
    done = {name: row for name, row in grid.items() if row.get("status") == "complete"}
    if not done:
        raise ValueError("no grid config completed")
    base_name = DEFAULT_CONFIG if DEFAULT_CONFIG in done else min(done, key=TIE_ORDER.index)
    base = done[base_name]
    qualifying = [
        name
        for name, row in done.items()
        if name != base_name
        and base["val_log_loss"] - row["val_log_loss"] >= MIN_IMPROVEMENT
        and row["val_accuracy"] >= base["val_accuracy"]
    ]
    if qualifying:
        best = min(row["val_log_loss"] for name, row in done.items() if name in qualifying)
        tied = [n for n in qualifying if done[n]["val_log_loss"] - best <= TIE_TOLERANCE]
        selected = min(tied, key=TIE_ORDER.index)
        reason = (
            f"{selected} beat {base_name} by {base['val_log_loss'] - done[selected]['val_log_loss']:.4f} "
            f"validation log-loss (margin {MIN_IMPROVEMENT}) without lower validation accuracy"
        )
    else:
        selected = base_name
        reason = f"no challenger beat {base_name} by {MIN_IMPROVEMENT} validation log-loss without lower accuracy"
    return {
        "selected": selected,
        "policy": done[selected]["policy"],
        "reason": reason,
        "scores": {n: {"val_log_loss": r["val_log_loss"], "val_accuracy": r["val_accuracy"], "policy": r["policy"]}
                   for n, r in done.items()},
        "fingerprint": done[selected].get("fingerprint"),
    }


def paired_bootstrap(
    selected: Sequence[Mapping[str, Any]],
    frozen: Sequence[Mapping[str, Any]],
    *,
    clusters: Sequence[Any] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired bootstrap of (selected - frozen probe), 95 % percentile CIs. Amendment 1 (2026-10-18): on the fresh
    test the resampling unit is the OBSERVER (`clusters`, one entry per image), so photographs by one observer move
    together. Accuracy difference = primary endpoint; log-loss difference = secondary, never part of the verdict.
    Without `clusters` every image is its own cluster (used only for the selection-time look on validation)."""
    if [r["id"] for r in selected] != [r["id"] for r in frozen]:
        raise ValueError("the two systems must be scored on the same images in the same order")
    d_acc = np.array([float(a["correct"]) - float(b["correct"]) for a, b in zip(selected, frozen, strict=True)])
    d_nll = np.array([a["nll"] - b["nll"] for a, b in zip(selected, frozen, strict=True)])
    labels = list(clusters) if clusters is not None else list(range(len(d_acc)))
    if len(labels) != len(d_acc):
        raise ValueError("one cluster label per image is required")
    order = sorted(set(labels), key=str)
    index = {c: i for i, c in enumerate(order)}
    cl = np.array([index[c] for c in labels])
    count = np.bincount(cl, minlength=len(order)).astype(float)
    sum_acc = np.bincount(cl, weights=d_acc, minlength=len(order))
    sum_nll = np.bincount(cl, weights=d_nll, minlength=len(order))
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(order), size=(n_boot, len(order)))
    n_draw = count[draw].sum(axis=1)
    acc = sum_acc[draw].sum(axis=1) / n_draw
    nll = sum_nll[draw].sum(axis=1) / n_draw
    return {
        "n_images": len(d_acc),
        "n_clusters": len(order),
        "resampling_unit": "observer" if clusters is not None else "image",
        "n_boot": n_boot,
        "seed": seed,
        "primary_accuracy_difference": float(d_acc.mean()),
        "primary_accuracy_ci95": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
        "discordant_images": {"selected_only": int((d_acc > 0).sum()), "probe_only": int((d_acc < 0).sum())},
        "secondary_log_loss_difference": float(d_nll.mean()),
        "secondary_log_loss_ci95": [float(np.percentile(nll, 2.5)), float(np.percentile(nll, 97.5))],
    }


def reading_verdict(boot: Mapping[str, Any], *, same_model: bool) -> str:
    """Amendment 1 verdict, from the primary accuracy CI only."""
    if same_model:
        return "same model"
    low, high = boot["primary_accuracy_ci95"]
    if low > 0:
        return "improved"
    if high < 0:
        return "worse"
    return "no detectable difference"


def scoring_mode(*, differences: Sequence[str], grid: Mapping[str, Any], state_matches: bool, masked: bool) -> str:
    """'preregistered' only when every check holds; 'masked' for a development check; else 'exploratory'."""
    if differences or not grid.get("complete") or not state_matches:
        return "exploratory"
    return "masked" if masked else "preregistered"


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write via a temporary file and os.replace: a reader never sees a half-written file."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json_or_none(path: str | Path) -> Any:
    """The parsed file, or None when it is missing, empty or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def compare_with_record(fresh_test: Mapping[str, Any], selected_config: str,
                        record: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """How a reproduction compares with the evaluation of record (printed and saved; it asserts nothing).

    None when there is no record or the fresh test was not scored. Values are compared at the record's 4-decimal
    precision; a hosted GPU may still differ in the last digit, which is why this is reported, not enforced.
    """
    record = RECORDED_EVALUATION if record is None else record
    if not record or not fresh_test.get("scored"):
        return None
    rec = record["fresh_test"]
    boot = fresh_test.get("paired_bootstrap") or {}
    r4 = lambda x: None if x is None else round(float(x), 4)  # noqa: E731
    return {
        "record": f"{record['date']} {record['commit'][:7]} / blob {record['notebook_blob'][:8]}",
        "selected_config": selected_config == record["selected_config"],
        "selected_accuracy": r4(fresh_test["selected"]["accuracy"]) == rec["selected"]["accuracy"],
        "probe_accuracy": r4(fresh_test["frozen_probe"]["accuracy"]) == rec["frozen_probe"]["accuracy"],
        "primary_accuracy_difference": r4(boot.get("primary_accuracy_difference")) == rec["primary_accuracy_difference"],
        "primary_accuracy_ci95": [r4(x) for x in boot.get("primary_accuracy_ci95") or []] == rec["primary_accuracy_ci95"],
        "discordant_images": boot.get("discordant_images") == rec["discordant_images"],
        "verdict": fresh_test.get("verdict") == rec["verdict"],
    }


class LookLedger:
    """Counts fresh-test scorings in this runtime, crash-safely (row 108 F2).

    A look is recorded only by `record()`, after the report is safely on disk; `touched()` marks a fetch that started.
    Whether the pre-registered slot is spent is decided from the outputs themselves as well as the ledger: a readable
    main report means every later scoring is a reproduction, even if `work/` was lost; a missing or unreadable main
    report is (re)written unless the ledger proves a completed look. A corrupt ledger reads as empty.
    """

    def __init__(self, path: str | Path, outputs_dir: str | Path = "outputs", stem: str = "vit_classification") -> None:
        self.path = Path(path)
        self.outputs = Path(outputs_dir)
        self.stem = stem

    def _load(self) -> dict[str, int]:
        data = read_json_or_none(self.path)
        if not isinstance(data, dict):
            return {}
        return {k: int(v) for k, v in data.items() if isinstance(v, int | float)}

    def _save(self, counts: Mapping[str, int]) -> None:
        atomic_write_text(self.path, json.dumps(dict(counts)))

    @property
    def main_report(self) -> Path:
        return self.outputs / f"{self.stem}_evaluation_report.json"

    def _highest(self, kind: str) -> int:
        import re

        pattern = re.compile(rf"^{re.escape(self.stem)}_evaluation_report_{kind}_(\d+)\.json$")
        found = [int(m.group(1)) for f in (self.outputs.glob("*.json") if self.outputs.is_dir() else [])
                 if (m := pattern.match(f.name))]
        return max(found, default=0)

    def plan(self, mode: str) -> tuple[str, str]:
        """(test_look, file suffix) for a scoring of kind `mode`. Records nothing."""
        if mode not in ("preregistered", "exploratory", "masked"):
            raise ValueError(f"unknown scoring mode {mode!r}")
        counts = self._load()
        if mode == "preregistered":
            # With the evaluation of record set, a pre-registered scoring in ANY runtime (a new one included) is a
            # reproduction and never writes the main files.
            if (RECORDED_EVALUATION is not None or read_json_or_none(self.main_report) is not None
                    or counts.get("preregistered_completed", 0) > 0):
                n = max(self._highest("reproduction"), counts.get("reproductions", 0)) + 1
                return TEST_LOOK_REPRODUCTION, f"_reproduction_{n}"
            return TEST_LOOK_PREREGISTERED, ""
        n = max(self._highest(mode), counts.get(mode, 0)) + 1
        return (TEST_LOOK_MASKED if mode == "masked" else TEST_LOOK_EXPLORATORY), f"_{mode}_{n}"

    def touched(self) -> int:
        """Mark that a fresh-test fetch started (an aborted scoring stays visible). Returns the running count."""
        counts = self._load()
        counts["fresh_test_touched"] = counts.get("fresh_test_touched", 0) + 1
        self._save(counts)
        return counts["fresh_test_touched"]

    def touches(self) -> int:
        return self._load().get("fresh_test_touched", 0)

    def record(self, mode: str, suffix: str) -> None:
        """Count a completed scoring; call it only after its report is on disk."""
        counts = self._load()
        if mode == "preregistered":
            if suffix == "":
                counts["preregistered_completed"] = counts.get("preregistered_completed", 0) + 1
            else:
                counts["reproductions"] = max(counts.get("reproductions", 0), int(suffix.rsplit("_", 1)[1]))
        else:
            counts[mode] = max(counts.get(mode, 0), int(suffix.rsplit("_", 1)[1]))
        self._save(counts)

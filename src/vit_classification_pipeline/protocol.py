"""The tutorial's pre-registered selection and fresh-test rules (docs/vit-preregistration.md, 2026-10-18).

`PREREGISTERED_SETTINGS` lists every setting the pre-registration fixes. A run is `preregistered` only when all
of them hold, the validation grid is complete and matches them, and the model in hand is the selected system;
only such a run scores the fresh test. Every other run is exploratory: validation and the labelled exposed
development test only, written to `*_exploratory_<n>` files. A repeat of a pre-registered scoring in the same
runtime is a reproduction and writes `*_reproduction_<n>` files; it never overwrites the first one. A runtime cannot
see other runtimes: docs/release-verification.md says which hosted run is the evaluation of record
(`RECORDED_EVALUATION`, None until that run exists).
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

# The hosted T4 run that is the evaluation of record (date, notebook blob, numbers); None until it exists.
RECORDED_EVALUATION: dict[str, Any] | None = None

TEST_LOOK_PREREGISTERED = "pre-registered evaluation (this runtime)"
TEST_LOOK_REPRODUCTION = "reproduction of the pre-registered evaluation"
TEST_LOOK_EXPLORATORY = "not scored (exploratory)"
TEST_LOOK_MASKED = "not scored (fresh test masked for a development check)"


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
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired bootstrap over images of (selected - frozen probe) in accuracy and log-loss, 95 % percentile CI."""
    if [r["id"] for r in selected] != [r["id"] for r in frozen]:
        raise ValueError("the two systems must be scored on the same images in the same order")
    d_acc = np.array([float(a["correct"]) - float(b["correct"]) for a, b in zip(selected, frozen, strict=True)])
    d_nll = np.array([a["nll"] - b["nll"] for a, b in zip(selected, frozen, strict=True)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d_acc), size=(n_boot, len(d_acc)))
    acc, nll = d_acc[idx].mean(axis=1), d_nll[idx].mean(axis=1)
    return {
        "n_images": len(d_acc),
        "n_boot": n_boot,
        "seed": seed,
        "accuracy_difference": float(d_acc.mean()),
        "accuracy_ci95": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
        "log_loss_difference": float(d_nll.mean()),
        "log_loss_ci95": [float(np.percentile(nll, 2.5)), float(np.percentile(nll, 97.5))],
    }


def reading_verdict(boot: Mapping[str, Any], *, same_model: bool) -> str:
    """The pre-registered verdict, read from the CIs only."""
    if same_model:
        return "same model (probe kept)"
    acc_lo, acc_hi = boot["accuracy_ci95"]
    nll_lo, nll_hi = boot["log_loss_ci95"]
    better = acc_lo > 0 or nll_hi < 0
    worse = acc_hi < 0 or nll_lo > 0
    if better and not worse:
        return "unfreeze better"
    if worse and not better:
        return "unfreeze worse"
    return "no detectable difference"


def scoring_mode(*, differences: Sequence[str], grid: Mapping[str, Any], state_matches: bool, masked: bool) -> str:
    """'preregistered' only when every check holds; 'masked' for a development check; else 'exploratory'."""
    if differences or not grid.get("complete") or not state_matches:
        return "exploratory"
    return "masked" if masked else "preregistered"


class LookLedger:
    """Counts scorings in this runtime (on disk, so re-running a cell or Section 4 does not reset it)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, int]:
        if self.path.is_file():
            return {k: int(v) for k, v in json.loads(self.path.read_text(encoding="utf-8")).items()}
        return {}

    def next_look(self, mode: str) -> tuple[str, str]:
        """(test_look, file suffix) for the next scoring of kind `mode`, and record it."""
        if mode not in ("preregistered", "exploratory", "masked"):
            raise ValueError(f"unknown scoring mode {mode!r}")
        counts = self._load()
        n = counts.get(mode, 0)
        counts[mode] = n + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(counts), encoding="utf-8")
        if mode == "preregistered":
            return (TEST_LOOK_PREREGISTERED, "") if n == 0 else (TEST_LOOK_REPRODUCTION, f"_reproduction_{n}")
        if mode == "masked":
            return TEST_LOOK_MASKED, f"_masked_{n + 1}"
        return TEST_LOOK_EXPLORATORY, f"_exploratory_{n + 1}"

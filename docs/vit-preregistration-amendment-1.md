# Pre-registration amendment 1 (2026-10-18): the fresh-test reading rule and look counting

Amends `docs/vit-preregistration.md` (commit `7ed4d60`). Committed on its own, before the code that implements it.
**No fresh-test score exists when this is written**: the fresh test has never been fetched or scored, by any run.
The original text is not edited; where they differ, this amendment wins.

## Why amend

The peer review of PR #14 (findings F2, F3, F5, F7, F9, F10) found:
- the original rule joined two 95 % tests (accuracy OR log-loss) and so had an error rate near 9-10 %;
- it resampled images as if they were independent, although the 96 photographs come from 79 observers;
- it labelled opposite intervals "no detectable difference";
- look counting was not crash-safe.

## What was known when this was written (disclosure)

A masked CPU development check of notebook blob `085cdcf7` (commit `9919e13`, 2026-10-18) gave these results on
validation:

| Config | Validation log-loss | Validation accuracy |
|---|---|---|
| C0 (keeps the probe) | 0.1621 | 22/24 |
| C1 (epoch 4) | 0.1242 | 23/24 |
| C2 (epoch 2) | 0.1375 | — |

The grid rule selected **C1**. On the exposed development test, C1 scored 72.9 % against the probe's 79.2 %
(3 photographs of 48).

The fresh test was masked in that run (`DIMER_MASK_FRESH_TEST=1`), so it was not fetched or scored. These numbers
changed no setting. They are the reason the text must cover the C1-selected case.

## 1. Primary endpoint and reading rule (replaces "Reading rule for the fresh test")

- **Primary endpoint.** One endpoint: the paired accuracy difference on the 96 fresh photographs, **selected system −
  frozen probe**.
- **Interval.** A 95 % percentile interval from a **cluster bootstrap resampled by observer**. There are 79 observer
  clusters, and all photographs of a resampled observer enter together. `n_boot` = 10,000, seed 20261017.
- **Verdict**, from the primary interval only:
  - "improved" if the lower bound is > 0;
  - "worse" if the upper bound is < 0;
  - "no detectable difference" otherwise;
  - "same model" when the selected system is the probe. Both differences are then exactly 0.
- **Secondary, never part of the verdict.** These are reported with their own observer-cluster intervals:
  - the log-loss difference;
  - the discordant counts;
  - each system's accuracy, macro-F1 and log-loss, beside the uniform-guess log-loss ln 6 = 1.792.
- **Power, stated in advance.** At n = 96 with 79 clusters, the expected half-width of the accuracy interval is
  about:
  - **±4.4 points** if 5 % of images are discordant;
  - **±6.4 points** at 10 %;
  - **±7.5 points** at 15 %.

  These figures come from a simulation of the committed sampler, 200 replicates per level. A difference of a few
  photographs therefore reads as "no detectable difference". The verdict concerns this one trained model (one
  seed-0 run per config) against this probe, not the unfreeze method in general.

## 2. Added reference systems on the fresh test (F10)

Under the same pre-registered gate and in the same look, two reference systems are added. They are descriptive and
not part of the verdict.
- The majority floor: 1/6 at 16 per species.
- The cosine 5-NN vote of the frozen base features, using the 108 training photographs.

## 3. Look counting (F2, F3, F9)

- **When a look counts.** A look counts only once its report is written. Reports and the look ledger are written
  atomically (temporary file, then rename).
  - A fetch that starts is logged as "touched". An aborted scoring is therefore visible, but is not a look.
  - If a readable main report already exists, every later scoring in that runtime is a reproduction, even if
    `work/` was lost.
- **Report fields.** Reports carry:
  - `preregistered_settings_hold` and `masked`;
  - `fresh_test_scored`;
  - `preregistered` (True only for an unmasked scoring under the pre-registered gate).
  - A reproduction also carries `matches_first`.
- **Masking.** `fetch_fresh_test` refuses while `DIMER_MASK_FRESH_TEST=1`, and CI sets that mask.
- **Hosted runs.** Every hosted run that reaches the fresh-test fetch is numbered in `docs/release-verification.md`
  (hosted look #1, #2, …). Hosted look #1, if it completes, is the evaluation of record.

## 4. Device and precision (F7)

- The evaluation of record is the first hosted **Tesla T4** run.
- TF32 is disabled in the pipeline (`cudnn.allow_tf32` and `cuda.matmul.allow_tf32` = False). Every report records:
  - the device and the GPU name;
  - the float32 matmul precision;
  - the TF32 flags.
- A pre-registered scoring on another device is labelled as such and is not the record. GPU training is not
  bit-deterministic, so it may train a different C1.

## 5. The split check (F15)

The grid rows and Section 8 hash the records actually trained on (`dataset_digest` of the train and validation
records). Section 8 compares those hashes with Section 4's manifests and with the grid rows. Any mismatch makes the
run exploratory.

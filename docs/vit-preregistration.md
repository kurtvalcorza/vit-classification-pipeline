# Pre-registration: validation grid and fresh test for the ViT tutorial

Committed on 2026-10-18, on its own, before any code that implements it and before any grid run by this fix.
Owner rulings R108-D1 (fresh test) and R108-D2 (grid), 2026-10-17. Later changes go in a dated amendment below;
this text is not edited.

## What was already known when this was written (disclosure)

- The build sweep of 2026-09-18/19 (CPU pre-flight of notebook blob `c1fdbadd`, PR #3) **scored two
  configurations on the original 48-photograph test split**: 4 blocks @ 1e-4 (selected on validation at epoch 4,
  72.9 % test accuracy) and 2 blocks @ 3e-4 (selected at epoch 2, 66.7 %). The default recipe (2 blocks @ 3e-5,
  4 epochs), the lesson "the probe is usually the honest choice — here it was" and the Section 10 default
  experiment were set with those test numbers in view. That split is therefore an **exposed development test**
  from now on.
- The build record's validation numbers for the default were known: its epochs never beat the probe's validation
  log-loss (0.162 against 0.163–0.166), and both C1 and C2 below beat the probe on validation in that sweep. The
  0.02 margins below were proposed by the reviewer with those values in view. They are practical tolerances,
  not significance thresholds.
- The provenance of `PROBE_STEPS` 300, `PROBE_LR` 1e-2 and `BATCH_SIZE` 8 is not recorded; they are held fixed.
- The fresh test below has never been scored by this repository's model. Its photographs are allocated in six
  other fleet repositories (siglip2, siglip-v1, lightglue, xoftr, swin2sr, vit-mae), whose models have scored
  them in their own splits.

## Fixed settings (all must hold for a run to be `preregistered`)

| Setting | Value |
|---|---|
| Data source | the pinned 180-photograph corpus (`SAMPLE_RECORDS`, 19,183,071 bytes), BYOD off |
| Split | `SPLIT_SEED` 42; 18 / 4 / 8 per species (108 / 24 / 48) |
| Linear probe | `PROBE_STEPS` 300, `PROBE_LR` 1e-2, full batch, AdamW |
| Unfreeze training | `EPOCHS` 4, `BATCH_SIZE` 8, adapt seed 0, weight decay 0.01, gradient clip 1.0, float32, no augmentation |
| Keep-frozen margin | `min_improvement` 0.02 inside `adapt` |
| Grid | C0, C1, C2 below, all three complete, in that order |
| Optional experiment | Section 10 off |
| Fresh test | the 96 records of `FRESH_TEST_RECORDS`, selection digest `01fffe006c19d873bdcb379e6345f3f8219490340a76ca8e42a68ff19e9dda84` (`docs/fresh-test-draw.json`) |

## Grid (validation only)

| Config | Trainable blocks | Learning rate |
|---|---|---|
| C0 (current default) | 2 | 3e-5 |
| C1 | 4 | 1e-4 |
| C2 | 2 | 3e-4 |

Each config runs `adapt` from the pinned base with every fixed setting above.

**Keep frozen unless beaten.** Inside one config, epoch 0 is the linear probe. The best unfrozen epoch is the one
with the lowest validation log-loss among epochs 1..4. It replaces the probe only if its validation log-loss is at
least 0.02 below the probe's (`min_improvement=0.02`); otherwise the config's selected system is the probe.

**Choosing the config.** The score of a config is the validation log-loss of its selected system. A challenger
(C1 or C2) replaces C0 only if its score is at least 0.02 below C0's **and** its validation accuracy is not lower
than C0's. If both qualify, the lower score wins. Ties (equal to 1e-9) go to C0, then to C2 (smaller trainable
scope) over C1. If every config keeps the probe, the selected system is the probe and C0 is reported.

**Incomplete grid.** A config that fails (for example out of memory) is recorded as failed; the grid is then
incomplete, the run is exploratory, and the fresh test is not scored.

The grid runs in the default path (estimated about 40 s on a T4, measured C0 11.0 s on T4).

## Fresh test (R108-D1)

- 16 photographs per species, 96 in total, 10,659,247 bytes, 79 distinct observers, drawn by
  `tools/draw_fresh_test.py` from the 360-photograph table of `kurtvalcorza/siglip2-vision-language-pipeline`
  (`src/siglip2_pipeline/samples.py` at `4fe334864548adcbc18d4173d0e1d38ce2991259`) minus the 180, observers absent
  from the 180 only, in ascending photo id. Each is pinned by photo id, byte size and SHA-256.
- Disjointness: no photograph, observation or observer is shared with the 180. Near-duplicate screen: 64-bit dHash
  at Hamming distance <= 10 plus an exact decoded-pixel digest, against the 180 and within the draw; 0 hits,
  0 skipped. The fresh test is therefore observer-disjoint from training, unlike the exposed development test.
- It is scored **only** when every fixed setting holds and the model state equals the selected system. Every other
  run is exploratory: validation and the labelled exposed development test only, written to
  `*_exploratory_<n>.json`.

## Reading rule for the fresh test (fixed before any fresh score)

- Report accuracy, macro-F1 and log-loss of the selected system and of the frozen probe on the 96 photographs.
- Paired bootstrap over images: `n_boot` 10,000, seed 20261017, resample the 96 images with replacement, compute
  the difference **selected − frozen probe** in accuracy and in log-loss, and take the 2.5 / 97.5 percentiles as the
  95 % CI.
- Verdict, from the CI only:
  - "same model (probe kept)" when the selected system is the probe (both differences are exactly 0);
  - "unfreeze better" when the accuracy CI lies above 0 **or** the log-loss CI lies below 0, and neither CI points
    the other way;
  - "unfreeze worse" when the accuracy CI lies below 0 **or** the log-loss CI lies above 0, and neither CI points
    the other way;
  - otherwise "no detectable difference".
- The exposed development test is reported once beside it, labelled as exposed, and never used for a verdict.

## Evaluation of record

The first hosted T4 Run all with every fixed setting is the evaluation of record. A later hosted run with identical
settings is a reproduction. Fixer CPU checks run with the fresh test masked (SCORE_ONCE_RULE item 8).

## Amendments

None.

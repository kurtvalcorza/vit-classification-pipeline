# Release verification

`tutorials/vit_classification_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 3-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the 180 photo
  digests live in the carried `samples.py`, not in prose);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ViTClassificationPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `class_names`, `check_split_disjoint`, `observer_overlap`, `write_dataset_csv`, `validate_inputs` with the
  oversized-image refusal probe, `pipe.predict` with the sanity checks and the three ImageNet top-5 lists, `majority_baseline`,
  `pipe.knn_baseline`, `pipe.adapt(trainable_blocks=0)` with its frozen-policy assertion, `pipe.evaluate` on the
  frozen policy and on the validation and test splits after the unfrozen policy with the floor assertion,
  `pipe.adapt` with `trainable_blocks=TRAINABLE_BLOCKS` and `lr=LEARNING_RATE`, `pipe.classify` before and after, the
  per-batch `evaluation_report`, `pipe.save_artifact`, `ViTClassificationPipeline.from_artifact` and the
  reload-parity assertion including `predict`, and the provenance fields `weight_format`, `weight_sha256` and the `corpus` block), the
  six expected `outputs/` paths, the learner-facing statements (the fixed ImageNet-1k label space, supervised
  adaptation to a label space its ImageNet-1k head does not have under an explicit frozen-vs-unfrozen policy, the majority
  floor, the cosine 5-NN vote, the frozen and unfrozen policies, lowest validation log-loss, the ImageNet head never
  trained or exported, no dispersion estimate, named exclusions, the CC0 licence) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git
  clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `timm.create_model(` / `from timm` / `from huggingface_hub import` / `urllib.request` / `safetensors` /
  `torch.optim` / `.backward(` / `pipe._model` / `cross_entropy(` / `torch.nn.Linear(` / `forward_features(` /
  `forward_head(` / `.logits` use **outside the carried
  module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`,
  `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `timm`, `huggingface-hub`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_notebook_parity.py`; injected zero-logit and mean-colour feature runners and corpus fetcher, synthetic JPEG swatches, temporary
manifests, no weights — `tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and
unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy, Pillow or torch that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/vit-base-p16-224/` or the photo cache `weights/inat-birds/` (the standalone path writes the
   manifest itself, stages the missing file from the Hub, and fetches the 180 pinned photographs from the
   iNaturalist open-data bucket, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `PROBE_STEPS = 300`, `PROBE_LR = 0.01`, `EPOCHS = 4`,
   `BATCH_SIZE = 8`, `RUN_GRID = True`; `DIMER_MASK_FRESH_TEST` unset). **Rows below that describe Sections 6-9
   predate the 2026-10-17 review fixes**: since then Section 7 runs the C0 / C1 / C2 grid and reloads the selected
   config (read it from the run, it is not pre-stated; a masked CPU check selected C1), refusals raise `RunStopped`,
   four modules are carried (`protocol.py` added), and Sections 8-9 write suffixed files (`…_evaluation_report.json`
   for the first pre-registered evaluation, `_reproduction_<n>` / `_exploratory_<n>` / `_masked_<n>` otherwise) plus
   `…_grid.json`; Section 8 asserts nothing about the floor. Check the report's `preregistered`, `test_look`,
   `settings_differing`, `numeric_settings` (device `cuda:0`, GPU `Tesla T4`, TF32 off) and the fresh-test verdict
   instead;
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0` (an interpreter restart after the install is expected
   where the runtime's preinstalled torch, numpy or Pillow differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `ViTClassificationPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `classification_metrics`, `majority_baseline`,
     `knn_predict`, `SAMPLE_RECORDS`, `SPECIES`, `fetch_corpus`, `read_corpus`, `build_sample_dataset`,
     `validate_dataset`, `class_names`, `check_split_disjoint`, `observer_overlap`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `timm/vit_base_patch16_224.orig_in21k_ft_in1k` at the immutable revision, and `verify_snapshot` returning its dict
     (3 files); `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source`
     `local-snapshot` at the fixed 224 × 224 input;
   - Section 4: `fetch_corpus` fetching the 180 pinned photographs (19,183,071 bytes) from
     `inaturalist-open-data.s3.amazonaws.com` into `weights/inat-birds/`, six species of 30 read, and the seeded
     stratified draw of 108 / 24 / 48 records with `check_split_disjoint` reporting no shared photograph, the observer
     overlap counted (31 of 117 observers in more than one split in the recorded run) and the three dataset digests
     `1e4cca7f…` / `d176b3ff…` / `0e787f09…`; `outputs/…_train.csv` written; the four dataset refusal probes each
     raising `ValueError`;
   - Section 5: the ceilings (`NUM_CLASSES` 1000, `MAX_IMAGE_SIDE` 4096, `MAX_BATCH` 64) and the contract (`INPUT_SIZE`
     224, `FEATURE_DIM` 768, `DECISION_RULE` `argmax`, `TRANSFORMER_BLOCKS` 12, `PARAMETER_COUNT` 86,567,656) surfaced;
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from
     the oversized probe); `predict` on three test photographs with all four sanity checks `True` and the three
     ImageNet-1k top-5 lists printed beside the gold species (a goldfinch named `goldfinch` at 0.999 in the recorded
     run; the two sparrows given ImageNet labels that do not exist for them);
   - Section 6: the majority floor (16.7 % accuracy), the cosine 5-NN vote (≈ 75.0 % on the sample), and the
     frozen policy — `adapt(trainable_blocks=0)` reporting `frozen backbone + linear probe` with its validation
     metrics — scored on the test split (≈ 79.2 % accuracy, macro-F1 ≈ 0.789) on CPU float32, with the
     cell's assertion that the probe beats the floor and the policy is the frozen one;
   - Section 7: `pipe.adapt` printing epoch 0 as the linear probe (validation log-loss ≈ 0.162), then
     4 unfreeze epochs of the last two blocks (14,175,744 trainable of 86,567,656 parameters plus
     the 4,614-parameter head) with validation log-loss / accuracy each epoch (0.162 (91.7 %) → 0.165 (91.7 %) → 0.166 (91.7 %) → 0.163 (91.7 %) → 0.166 (91.7 %) in the recorded run) and
     the selected policy `frozen backbone + linear probe` (`best_epoch` 0 — the unfreeze never beat the probe);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, per-class recall and
     the confusion matrix, and `outputs/…_evaluation_report.json` written (the cell asserts the selected model beats
     the majority floor — on the sample ≈ 79.2 % versus 16.7 %; the delta over the probe, +0.0
     points because the probe was selected, is reported, not asserted);
   - Section 9: six test photographs labelled by the selected model and by a fresh frozen-policy probe, printed with
     the gold species and top probabilities, the per-batch `evaluation_report` verdict `not-measurable`,
     `outputs/…_predictions.csv` written; `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,
     manifest.json}` (2 tensors, 18,632 B when the probe is selected — 26 tensors, about 56.7 MB when two blocks are
     trained; `classes` and `policy` recorded; never the ImageNet head) and `ViTClassificationPipeline.from_artifact`
     reloading it with identical probabilities on the six images, an identical test accuracy and identical `predict`
     output on the three probe photographs (the cell asserts all three); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the
     model identity and licence, the snapshot block (`weight_format`, `weight_sha256`), the `corpus` block, the
     inference-contract items, the comparison, the before/after predictions, the artifact digest and policy, the
     reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, timm, device), the model identifier
   and immutable revision, whether the model cache, the weights directory and the photo cache were clean, outcome,
   produced outputs, the observed metrics and the selected policy (as observations, not a benchmark) and any warning
   or applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `vit_classification_colab.ipynb` (`E2E`) | `e96ee0e` / `bb9441e4` | 2026-10-18 | Colab CLI 0.7.4, fresh Colab Tesla T4 (CLI sequential execution, not a browser Run all; forms and uploads not exercised) | **PASSED** — 14/14 code cells, 0 errors, 262.5 s; **reproduction of the pre-registered evaluation** (REL13 label; reproduction 1 of the `1231036` / `94e67934` record): `test_look` `reproduction`, `matches_record` True on all seven fields; C1 selected; fresh test C1 0.8125 against the probe 0.8333, difference −0.0208, CI [−0.0632, 0.0204], `no detectable difference`, equal to the record; only `_reproduction_1` files written; evidence `execution-evidence/2026-10-18-e96ee0e/`; status stays Candidate |
| `vit_classification_colab.ipynb` (`E2E`) | `1231036` / `94e67934` | 2026-10-18 | Colab CLI 0.7.4, fresh Colab Tesla T4 (CLI sequential execution, not a browser Run all; forms and uploads not exercised) | **PASSED** — 14/14 code cells, 0 errors, 317.0 s; **pre-registered evaluation of record** (REL13 label; first complete hosted run of the pre-registered settings and amendment 1, first scoring of the 96-photograph fresh test by any run): C1 selected; fresh test C1 0.8125 against the probe 0.8333, difference −0.0208, CI [−0.0632, 0.0204], `no detectable difference`; evidence `execution-evidence/2026-10-18-1231036/`; status stays Candidate |
| `vit_classification_colab.ipynb` (`E2E`) | `8b26ec6` / `50462c07` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-vit-classification` v4; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers ?` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 188 files, 365 MB staged from the Hub into a clean cache; comparison {accuracy: {majority_floor: 0.1667, knn5: 0.75, frozen_policy: 0.7917, selected_policy: 0.7917}, macro_f1: {majority_floor: 0.0476, knn5: 0.7516, frozen_policy: 0.7885, selected_policy: 0.7885}, log_loss: {frozen_policy: 0.678, selected_policy: 0.678}, per_class_recall: {american_goldfinch: {knn5: 0.88, frozen: 0.88, selected: 0.88}, chipping_sparrow: {knn5: 0.75, frozen: 0.88, selected: 0.88}, dark_eyed_junco: {knn5: 0.62, frozen: 0.75, selected: 0.75}, house_finch: {knn5: 0.88, frozen: 0.88, selected: 0.88}, song_sparrow: {knn5: 0.5, frozen: 0.5, selected: 0.5}, white_throated_sparrow: {knn5: 0.88, frozen: 0.88, selected: 0.88}}, delta_vs_frozen: {accuracy: 0, macro_f1: 0}, selected_policy: frozen backbone + linear probe}; reload parity {probabilities_identical: True, accuracy_in_memory: 0.7917, accuracy_reloaded: 0.7917, classes_identical: True, predict_identical: True}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-vit-classification/v4/evidence/` in the workspace |
| `vit_classification_colab.ipynb` (`E2E`) | `9f24c9e` / `c1fdbadd` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `vit_classification_colab.ipynb` (`TASK-INFERENCE`, superseded) | `83067a1` / `5b786a1e5767` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-vit-classification` v1) | PASSED — 8/8 code cells, 435.0 s, 346 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/vit_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/vit_classification_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-18 (08:56:10 UTC start) | `e96ee0e` / `bb9441e4542c` (`NOTEBOOK_SOURCE.repository_revision` `f6b5b3a`, equal to `metadata.dimer.generated_from.revision`; `f6b5b3a..e96ee0e` only regenerates the notebook; embedded `module_sha256` `63e9daad404e…`, generator `build_notebook.py/2.3`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 VM (session `suite-vit-e96ee0e-7e62`, `Hardware: T4`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; kernel Python 3.13.15; isolated `uv` environment CPython 3.12.12, 45 locked packages, built in 48 s; device `cuda:0` | Default pre-registered path, every form field at its default (BYOD off, `RUN_EXPERIMENT = False`); `preregistered` True, `settings_differing` []; snapshot `e0bd370d` 3 files digest-verified; 180 corpus photographs and the 96 fresh-test photographs fetched | 262.5 s (suite wall time, session request to stop) | **PASSED — one pass, no restart, 0 errors**: 14/14 code cells in order (`exec.log`); **reproduction of the pre-registered evaluation** (REL13; reproduction 1): `test_look` `reproduction`, `matches_record` True for selected config, selected and probe accuracy, primary difference, CI, discordant counts and verdict; grid C0 0.1621 / C1 0.1242 / C2 0.1375, **C1 selected**; fresh test C1 0.8125 (78/96) against the probe 0.8333 (80/96), −0.0208, CI [−0.0632, 0.0204], discordant 1 vs 3, `no detectable difference` — every printed result line equal to the record; files `*_reproduction_1` only. Evidence: `execution-evidence/2026-10-18-e96ee0e/` — executed notebook sha256 `417d78eec00f…`, `exec.log` sha256 `76d439cdf625…` (redacted; captured `8840ee4dc60c…`), `run_summary.json` sha256 `62b7c3a8d15b…`. BYOD and the Section 10 experiment not exercised; not promotion evidence on its own (REL12 pending) |
| 2026-10-18 (07:26:15 UTC start) | `1231036` / `94e67934a115` (`NOTEBOOK_SOURCE.repository_revision` `e0d4282`, equal to `metadata.dimer.generated_from.revision`; `e0d4282..1231036` only regenerates the notebook; embedded `module_sha256` `3713487a3a2e…`, generator `build_notebook.py/2.3`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 VM (session `suite-vit-1231036-7e62`, `Hardware: T4`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; kernel Python 3.13.15; isolated `uv` environment CPython 3.12.12, 45 locked packages, built in 55 s; torch 2.14.0+cu130, timm 1.0.29; device `cuda:0`; TF32 off in the carried code (see the record section) | Default pre-registered path, every form field at its default (BYOD off, `RUN_EXPERIMENT = False`); `preregistered` True, `settings_differing` []; snapshot `e0bd370d` 3 files digest-verified; 180 corpus photographs and the 96 fresh-test photographs fetched | 317.0 s (suite wall time, session request to stop) | **PASSED — one pass, no restart, 0 errors**: 14/14 code cells in order (`exec.log`); **pre-registered evaluation of record** (REL13): grid C0 0.1621 (probe kept) / C1 0.1242 (epoch 4) / C2 0.1375 (epoch 2), **C1 selected**; fresh test (n = 96) floor 0.1667, 5-NN 0.8542, frozen probe 0.8333, C1 0.8125; primary difference −0.0208, observer-cluster 95 % CI [−0.0632, 0.0204], discordant 1 vs 3, verdict `no detectable difference`, so **the adapted C1 did not beat the frozen probe**; exposed development test (decides nothing) probe 0.7917, C1 0.7292; reload parity identical (0.729167). Evidence: `execution-evidence/2026-10-18-1231036/` — executed notebook sha256 `f43bd2c13cbf…`, `exec.log` sha256 `e982129a47da…` (redacted; captured `65a0c96e15a7…`), `run_summary.json` sha256 `ca464c18db39…`. BYOD and the Section 10 experiment not exercised; not promotion evidence on its own (REL12 pending) |
| 2026-10-09 (14:50:37 UTC start) | `2351f94` / `5738b645259c` (`NOTEBOOK_SOURCE.repository_revision` `7086c87`, equal to `metadata.dimer.generated_from.revision`; `7086c87..2351f94` adds only the worker's google.colab ModuleSpec stubs and the spec 2.2 declaration; embedded `module_sha256` `f02fdc7b582e…`, generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 VM (session `suite-vit-2351f94-7e62`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; kernel Python 3.13.15; isolated `uv` environment CPython 3.12.12, 45 locked packages, built in 52 s; torch 2.14.0+cu130, timm 1.0.29; device `cuda:0` | Default sample path, every form field at its default (BYOD off, `PROBE_STEPS = 300`, `EPOCHS = 4`, `RUN_EXPERIMENT = False`); snapshot `e0bd370d` 3 files digest-verified; 180 photographs fetched and digest-verified | 185.4 s | **PASSED — one pass, no restart, 0 errors**: 13/13 code cells in order (`exec.log`); code cells 3–5 (the carried modules) print nothing; comparison {accuracy: {majority_floor: 0.1667, knn5: 0.75, frozen_policy: 0.7917, selected_policy: 0.7917}, macro_f1: {majority_floor: 0.0476, knn5: 0.7516, frozen_policy: 0.7885, selected_policy: 0.7885}}; validation kept the probe (epoch 0 log-loss 0.1621; unfrozen epochs 0.1634–0.166); verdicts above floor / above floor / no gain; song sparrow recall 0.5; reload parity identical (0.791667). Evidence: `execution-evidence/2026-10-09-2351f94/` — executed notebook sha256 `d1b8abc59ccd…`, `exec.log` sha256 `67cf7f21a7cc…` (redacted 2026-10-18, now `dec94e0cce44…`; see R108-S1 below), `run_summary.json` sha256 `bfbb778919e4…`. Optional journeys (BYOD, Section 10 experiment) not exercised; not promotion evidence on its own (REL12 pending) |
| 2026-09-19 | `8b26ec6` / `50462c07` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-vit-classification` v4; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers ?` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 276.0 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 188 files, 365 MB staged from the Hub into a clean cache; comparison {accuracy: {majority_floor: 0.1667, knn5: 0.75, frozen_policy: 0.7917, selected_policy: 0.7917}, macro_f1: {majority_floor: 0.0476, knn5: 0.7516, frozen_policy: 0.7885, selected_policy: 0.7885}, log_loss: {frozen_policy: 0.678, selected_policy: 0.678}, per_class_recall: {american_goldfinch: {knn5: 0.88, frozen: 0.88, selected: 0.88}, chipping_sparrow: {knn5: 0.75, frozen: 0.88, selected: 0.88}, dark_eyed_junco: {knn5: 0.62, frozen: 0.75, selected: 0.75}, house_finch: {knn5: 0.88, frozen: 0.88, selected: 0.88}, song_sparrow: {knn5: 0.5, frozen: 0.5, selected: 0.5}, white_throated_sparrow: {knn5: 0.88, frozen: 0.88, selected: 0.88}}, delta_vs_frozen: {accuracy: 0, macro_f1: 0}, selected_policy: frozen backbone + linear probe}; reload parity {probabilities_identical: True, accuracy_in_memory: 0.7917, accuracy_reloaded: 0.7917, classes_identical: True, predict_identical: True}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-vit-classification/v4/evidence/` in the workspace |
| 2026-09-19 | `9f24c9e` / `c1fdbadd` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `timm 1.0.29`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 3 entries because the snapshot was pre-staged → `verify_snapshot` 3 files → `from_pretrained` on CPU at 224 × 224 → `fetch_corpus` served from the pre-staged cache after its 180 digest checks (0.3 s) → six species of 30 read, 108 / 24 / 48 drawn with `check_split_disjoint` clean, 31 of 117 observers in more than one split, digests `1e4cca7f…` / `d176b3ff…` / `0e787f09…` → four dataset refusals → input manifest with the oversized-image refusal → `predict` of three test photographs (0.11 s) with all four sanity checks `True` and the ImageNet top-5 lists printed (chipping sparrow → `ruffed grouse` 0.261; goldfinch → `goldfinch` 0.999; song sparrow → `brambling` 0.366) → majority floor → 5-NN vote → frozen-policy probe → unfrozen-policy `adapt` → validation + test evaluation → predictions before/after (a fresh probe pipeline for the frozen column) → adapter export → reload parity incl. `predict`) | 74.7 s | **PASSED** — 11/11 code cells; majority floor 16.7 % / macro-F1 0.048; cosine 5-NN 75.0 % / 0.752 (6.9 s); frozen policy (linear probe, 300 steps, 7.9 s incl. features) 79.2 % / 0.789, log-loss 0.678, validation 91.7 % / log-loss 0.162; `adapt` with the unfreeze: 14,175,744 block + 4,614 head parameters, 4 epochs, 38.5 s, validation log-loss 0.162 (probe) → 0.165 → 0.166 → 0.163 → 0.166 with accuracy 91.7 % throughout, selected `frozen backbone + linear probe` at `best_epoch` 0; **selected policy on the test split 79.2 % / macro-F1 0.789, log-loss 0.678 (Δ +0.0 accuracy, +0.0 macro-F1 vs the probe — the probe is the selected model)**; per-class recall 0.88 / 0.88 / 0.75 / 0.88 / 0.50 / 0.88 (goldfinch, chipping sparrow, junco, house finch, song sparrow, white-throated sparrow); six predictions printed — none changed (`test-005`, a chipping sparrow, labelled house finch by both); per-batch report `not-measurable`; adapter 18,632 B / 2 tensors, SHA-256 `d216f083…`; reload parity exact (probabilities identical, test accuracy 0.791667 both ways, `predict` identical); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `83067a1` / `5b786a1e5767` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-vit-classification` v1) | Default sample path of the inference-only notebook: one synthetic gradient image, `stage_missing_files` fetching the three manifest entries from the Hub, `verify_snapshot` over 3 files, `predict` top-5 with the ceilings printed, `not-measurable` report, CSV + JSON exports | 435.0 s | **PASSED** — 8/8 code cells, 346 MB staged; history only |

### Multi-model image classification workshop

Notebook identity is the Git blob id of `tutorials/DIMER_MultiModel_Image_Classification_Workshop.ipynb`
(verify with `git rev-parse <commit>:tutorials/DIMER_MultiModel_Image_Classification_Workshop.ipynb`). This
notebook is verified separately from the primary tutorial above; each saved Colab copy was compared cell by cell
with the committed blob before it was recorded here.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-26 | `b9b94c9` / `5874024ac9b8` | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `STANDARD` tier (MobileNetV4-Conv-Small, ResNet-50, ConvNeXt-Tiny, ViT-B/16), built-in corpus, notebook unmodified | not recorded | **PASSED** — 23/23 code cells executed without error; 108 / 24 / 48 split, dataset digest `176d8cfa4d9f…`; test accuracy / log-loss MobileNetV4 0.521 / 2.70, ResNet-50 0.500 / 4.30, ConvNeXt-Tiny 0.729 / 1.49, ViT-B/16 0.792 / 0.98; reload parity 0.0 for every adapter; report bundle SHA-256 `45791b13036d…`. The run exposed a probe-recipe flaw: at learning rate `1e-2` every probe was selected at its first checkpoint (step 10) and ResNet-50's validation log-loss was 2.22, worse than a uniform six-way guess. Fixed in `e9237d0` (spec §17–18) |
| 2026-09-26 | `e9237d0` / `c40bc9351916` | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `STANDARD` tier, built-in corpus, notebook unmodified | not recorded | **PASSED** — 23/23 code cells executed without error; probes selected at steps 50 / 30 / 1000 / 1000 (none at the first checkpoint; ConvNeXt-Tiny and ViT-B/16 at the step cap); test accuracy / macro-F1 / log-loss MobileNetV4 0.500 / 0.504 / 1.52, ResNet-50 0.500 / 0.480 / 1.78, ConvNeXt-Tiny 0.708 / 0.710 / 1.17, ViT-B/16 0.771 / 0.770 / 0.89 (identical to the local CPU pre-flight to at least five decimals); reload parity 0.0 for every adapter; report bundle SHA-256 `b62c47e16c26…` |
| 2026-09-28 | `1ef3d0f` / `2cccb9c64fdb` (revision `0.2.0-candidate`) | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `STANDARD` tier, built-in corpus, notebook unmodified, activity comparison left at its default (skipped) | not recorded | **PASSED** — 24/24 code cells executed in order without error; 108 / 24 / 48 split, dataset digest `176d8cfa4d9f…`, 31 of 117 observers in more than one split; probes selected at steps 50 / 30 / 1000 / 1000; validation accuracy 0.625 / 0.708 / 0.958 / 0.958; test accuracy / macro-F1 / log-loss MobileNetV4 0.500 / 0.504 / 1.52, ResNet-50 0.500 / 0.480 / 1.78, ConvNeXt-Tiny 0.708 / 0.710 / 1.17, ViT-B/16 0.771 / 0.770 / 0.89 (identical to the `e9237d0` run), each scored over n = 48; majority floor 0.167; disagreement table over all 48 test images (15 unanimous correct, 5 unanimous wrong, 32 disagreements); reload parity 0.0 for every adapter; experiment `20260928T020445Z-6f8082`, 50 files in the report, bundle SHA-256 `f8aa524b5c50…` |
| 2026-10-03 | `e8d1dea` / `86f36b4733cc` (revision `0.3.0-candidate`) | Colab CLI 0.7.4, fresh Colab Tesla T4 (host Python 3.13.15; model environments CPython 3.12.12 from the hash locks) | `STANDARD` tier, built-in corpus, notebook unmodified, default path only; CLI sequential execution (`colab exec -f`), not a browser Run all | 204.6 s (whole run, incl. environment builds and downloads) | **PASSED** — 25/25 code cells executed in order without error; 108 / 24 / 48 split, dataset digest `176d8cfa4d9f…`, 31 of 117 observers in more than one split; probes selected at steps 50 / 30 / 1000 / 1000; validation accuracy 0.625 / 0.708 / 0.958 / 0.958; test accuracy / macro-F1 / log-loss MobileNetV4 0.500 / 0.504 / 1.52, ResNet-50 0.500 / 0.480 / 1.78, ConvNeXt-Tiny 0.708 / 0.710 / 1.17, ViT-B/16 0.771 / 0.770 / 0.89 (identical to the 2026-09-28 run), each scored over n = 48; majority floor 0.167; 15 unanimous correct, 5 unanimous wrong, 32 disagreements; reload parity 0.0 for every adapter; experiment `20261003T231450Z-b88746`, 50 files in the report, bundle SHA-256 `56d0b7cf951f…` |
| 2026-10-15 | `ed4b542` / `be5203b841ea` (revision `0.4.0-candidate`) | Colab CLI 0.7.4, fresh Colab Tesla T4 (host Python 3.13.15; model environments CPython 3.12.12 from the hash locks) | `STANDARD` tier plus the two ImageNet-1k-only counterparts (on by default), built-in corpus, notebook unmodified, default path only; CLI sequential execution (`colab exec -f`), not a browser Run all | 234.0 s (whole run, incl. environment builds and downloads) | **PASSED** — 27/27 code cells executed in order without error; 108 / 24 / 48 split, dataset digest `176d8cfa4d9f…`, 31 of 117 observers in more than one split; core probes selected at steps 50 / 30 / 1000 / 1000, test accuracy / macro-F1 / log-loss MobileNetV4 0.500 / 0.504 / 1.52, ResNet-50 0.500 / 0.480 / 1.78, ConvNeXt-Tiny 0.708 / 0.710 / 1.17, ViT-B/16 0.771 / 0.770 / 0.89 (identical to the 2026-10-03 run); ImageNet-1k-only counterparts (first run): ConvNeXt-Tiny `fb_in1k` selected at step 40, test 27/48 = 0.562 (nominal Wilson 95% [0.423, 0.693]) / macro-F1 0.556 / log-loss 1.13, ViT-B/16 `augreg_in1k` selected at step 50, test 22/48 = 0.458 [0.326, 0.597] / 0.459 / 1.48; planned paired contrasts (nominal exact McNemar): ConvNeXt-Tiny in12k vs in1k 8 vs 1 discordant, p = 0.0391; ViT-B/16 in21k vs in1k 16 vs 1, p = 0.0003; majority floor 0.167; 15 unanimous correct, 5 unanimous wrong, 32 disagreements (core models); reload parity 0.0 for all six adapters; experiment `20261015T015616Z-b1127b`, 67 files in the report, bundle SHA-256 `9fc4c6b56dff…` |

Notebook blob `2cccb9c64fdb` (revision `0.2.0-candidate`, from `1ef3d0f`) has a passing Google Colab T4 run of the default `STANDARD` path; the earlier blob `c40bc9351916` (from `e9237d0`) also passed. The
`FULL` tier (SwinV2-Tiny, EVA-02 Base 448) and the optional DINOv2 baseline have not been exercised on a hosted
runtime. As for the other notebooks, this table is the evidence record: `metadata.dimer.clean_runtime_evidence`
stays `pending` as authored, because editing it would change the blob these runs verify, and the registry status
stays **Candidate** until a reviewer promotes it.

## Primary ViT tutorial status

**Candidate.** The 2026-09-19 Kaggle T4 run of blob `50462c07` (commit `8b26ec6`) needed a manual restart after the install cell (11/11 ok (1 restart after install cell)), so it is not a one-pass `Run all` and not promotion evidence (review finding VIT-M1, 2026-10-05). The review-fix blob `5738b645259c` (commit `2351f94`; uv isolated environment, guided layer, VIT-M1..M3 / VIT-m1..m3 fixes) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-09 (Colab CLI 0.7.4 sequential execution, 13/13 code cells, 185.4 s; floor 0.1667, 5-NN 0.75, frozen probe 0.7917, selected policy 0.7917 with the probe kept on validation log-loss 0.1621; reload parity identical). On 2026-10-18 the review-fix blob `94e67934` (commit `1231036`) completed one pass with 0 errors on a fresh Colab Tesla T4 (Colab CLI 0.7.4 sequential execution, 14/14 code cells, 317.0 s): the pre-registered grid selected C1 (four blocks at 1e-4; validation log-loss 0.1242 against the probe's 0.1621), and on the 96-photograph fresh test, in this evaluation of record, **the adapted C1 did not beat the frozen probe**: 81.25 % (78/96) against 83.33 % (80/96), paired difference −0.0208, observer-cluster 95 % CI [−0.0632, 0.0204], 1 against 3 discordant photographs, verdict `no detectable difference`; record in "Primary ViT tutorial: hosted T4 evaluation of record (2026-10-18)" below. The regenerated blob `bb9441e4` (commit `e96ee0e`, carrying the record as `RECORDED_EVALUATION`) then ran the same pre-registered path on a new fresh Colab T4 (14/14, 0 errors, 262.5 s) as **reproduction 1 of the pre-registered evaluation**: every fresh-test number equals the record and only `_reproduction_1` files were written; see "Primary ViT tutorial: hosted T4 reproduction 1 (2026-10-18)". REL12 BYOD exercise (path-based and upload) and the optional Section 10 experiment are pending on a hosted runtime, so the status stays Candidate (REL14). The 2026-09-19 row and the local pre-flight rows above remain history. Any later change to the carried modules or to the notebook produces a new blob that needs its own clean run.

The current blob `dc2bae0b` (code commit `b5f6285`) shows a refused BYOD input as exactly one `Refused: <file>: <reason>` line (previously a differently prefixed line), with no traceback, and stops Run all at that cell; genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20). Its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD=True;BYOD_PATH='/etc/hostname'`) are pending, and the hosted runs of earlier blobs, including `bb9441e4`, are history for it (REL14).

## Primary ViT tutorial: 2026-10-17 review fixes (row 108)

Review `row108-vit` (2026-10-17) found that the default recipe and the lesson "the probe is usually the honest
choice" were set after the build sweep of 2026-09-18/19 (CPU pre-flight of blob `c1fdbadd`) had scored two
unfreeze configurations on the 48-photograph test split: 4 blocks @ 1e-4 (72.9 %) and 2 blocks @ 3e-4 (66.7 %).
That split is now the **exposed development test**. Every number on it in this file (including the 2026-10-09
row above) is an exposed number, reported for continuity.

- **Pre-registration** (`docs/vit-preregistration.md`, committed alone before any code): the C0 / C1 / C2
  validation grid, every fixed setting, the 0.02 log-loss margin with the keep-frozen rule, and the fresh-test
  reading rule (paired bootstrap of selected − probe, 10,000 resamples, seed 20261017, verdict from the 95 % CIs).
- **Fresh test** (owner ruling R108-D1): 96 photographs (16 per species), 10,659,247 bytes, 79 observers, selection
  digest `01fffe006c19d873bdcb379e6345f3f8219490340a76ca8e42a68ff19e9dda84`, drawn by `tools/draw_fresh_test.py`
  from the 360-photograph table of `kurtvalcorza/siglip2-vision-language-pipeline` at
  `4fe334864548adcbc18d4173d0e1d38ce2991259` minus the 180, observers absent from the 180, ascending photo id.
  0 shared photographs, observations or observers; 0 near-duplicate hits (dHash-64 <= 10 or equal pixels).
  This model has never scored them; six other fleet repositories allocate them in their own splits.
- **Score-once wiring** (`protocol.py`, SCORE_ONCE_RULE items 1-8): Section 8 re-checks `preregistered` over all
  fixed settings, the grid's completeness and settings, the split and the model-state fingerprint before it
  touches the fresh test. Reproductions write `_reproduction_<n>`, exploratory runs `_exploratory_<n>`.
- **Evaluation of record:** the 2026-10-18 Colab CLI T4 run of commit `1231036` / blob `94e67934` (hosted run #3
  below; section "Primary ViT tutorial: hosted T4 evaluation of record (2026-10-18)"). Later identical hosted
  runs are reproductions. **Reproductions:** 1 — the 2026-10-18 Colab CLI T4 run of commit `e96ee0e` / blob
  `bb9441e4` (hosted run #4 below; section "Primary ViT tutorial: hosted T4 reproduction 1 (2026-10-18)"), equal
  to the record. `RECORDED_EVALUATION` in `protocol.py` now carries this
  record (re-derived from the executed notebook; `test_recorded_evaluation_equals_the_executed_evidence`), and the
  notebook was regenerated from it (requote round, 2026-10-18; blob `bb9441e4`, executed as reproduction 1). From that blob on, a
  pre-registered run in any runtime, a new one included, prints `test_look` `reproduction`, writes only
  `*_reproduction_<n>` files and prints `matches_record`; the main report is never written again.
- **Development looks at the fresh test:** none. The fixer's CPU checks ran with `DIMER_MASK_FRESH_TEST=1`
  (validation and the exposed development test only).
- **Masked CPU development check, 2026-10-18** (commit `9919e13`, blob `085cdcf7`; Windows CPU float32, fresh test
  masked, nothing changed in response): grid validation log-loss C0 0.1621 (probe kept), C1 0.1242 (epoch 4),
  C2 0.1375 (epoch 2); **C1 selected** (beats C0 by 0.0379); exposed development test C1 72.9 % / macro-F1 0.718
  against the probe's 79.2 % / 0.789 (3 photographs of 48). Disclosed in amendment 1 before its code.
- **Amendment 1** (`docs/vit-preregistration-amendment-1.md`, committed alone before its code): one primary endpoint
  (paired accuracy difference, observer-cluster bootstrap), crash-safe look counting, TF32 off, the T4 run as record.
- **Hosted looks at the fresh test** (every unmasked hosted run that reaches the fetch is numbered here, whether or
  not it completes; the first COMPLETE one is the evaluation of record):
  - **Hosted fetch attempt #1, aborted, 0 fresh photographs scored** — 2026-10-18 ~07:00 UTC, Colab CLI T4, commit
    `7203721`, blob `793330fa`. Section 8 printed `preregistered` True, a complete grid, the model state matching and
    **C1 selected** (equal to the masked CPU check: validation log-loss C0 0.1621 / C1 0.1242 / C2 0.1375), and the
    exposed-test descriptives (C1 0.7292 against the probe's 0.7917). The fresh-test fetch then raised
    `ValueError: unsupported photo extension 'png'` in `photo_url` before any photograph was downloaded or decoded,
    so no fresh-test number exists and, by the amended look rule, this is not a look; Section 9 then failed with a
    `NameError` (no Section 8 report). Fix (round 3): `photo_url` accepts every served extension in the pinned
    tables (fresh: jpg 48, jpeg 45, png 2, JPG 1; corpus: jpg 93, jpeg 87), with the size and SHA-256 checks
    unchanged; a CPU test resolves every pinned URL without downloading; Section 9 stops with `RunStopped` when
    Section 8 did not complete. No pre-registered setting changed, so no amendment was needed.
  - **Hosted fetch attempt #2, aborted, 0 fresh photographs scored** — 2026-10-18 07:22:10–07:26:07 UTC, Colab CLI
    T4 (a new session `suite-vit-7203721-7e62`), the same commit `7203721` / blob `793330fa`, queued again by an
    orchestrator re-queue error after the round-3 fix existed. It printed the same Section 8 lines as attempt #1
    (`preregistered` True, grid complete, C1 selected with C0 0.1621 / C1 0.1242 / C2 0.1375, exposed-test
    descriptives C1 0.7292 against 0.7917), then raised the same `ValueError: unsupported photo extension 'png'`
    at the fresh-test fetch before any photograph was downloaded, and Section 9 failed with the same `NameError`.
    Not a look. Its printed output equals attempt #1 apart from process ids and timings. The executed notebooks
    of attempts #1 and #2 are not committed; both suite logs are kept in the maintainer's workspace.
  - **Hosted run #3, complete: the evaluation of record** — 2026-10-18 07:26:15–07:31:30 UTC, commit `1231036` /
    blob `94e67934`; the section below.
  - **Hosted run #4, complete: reproduction 1** — 2026-10-18 08:56:10–09:00:32 UTC, commit `e96ee0e` / blob
    `bb9441e4`, the same pre-registered settings, nothing changed in response to the test number; it scored the
    fresh test again and every result equals the record (section "Primary ViT tutorial: hosted T4 reproduction 1
    (2026-10-18)").
- **Evidence-log redaction (R108-S1):** `docs/execution-evidence/2026-10-09-2351f94/exec.log` lines 1 and 70
  named a local run directory; it is replaced by `<run-dir>/` (2 occurrences, nothing else changed). SHA-256
  before `67cf7f21a7cce686b8aca64c74d019d66196d78c06c6485126dc33d2bfbc7e33`, after
  `dec94e0cce4460918d0ad068a758c4e7a98dae262a8095251281a16c2fea03eb`. The executed notebook and
  `run_summary.json` of that run are unchanged.

## Primary ViT tutorial: hosted T4 evaluation of record (2026-10-18)

**Label (REL13): pre-registered evaluation of record.** It is the first complete hosted run of the pre-registered
settings and amendment 1, and the first time any run scored the 96-photograph fresh test. The fixer's CPU checks ran
with the fresh test masked, so there was no development look. The two aborted hosted attempts before it (listed
above) scored 0 fresh photographs.

- **Identity.** Commit `12310362e09bcf47dda3000266a52c072c91f891`, notebook blob
  `94e67934a115a3a3100c160681f7493753145a6d`; the run's `source.ipynb` hashes to that blob and is byte-equal to the
  committed notebook. `inspect_run.py`: 39/39 cells, same ids and order, source identity OK; embedded revision
  `e0d4282`, generator `build_notebook.py/2.3`, `module_sha256` `3713487a3a2e…`, spec 2.5. The CLI writes no execution
  counts; 10 of 14 code cells have output (the carried-module cells print nothing).
- **Runtime.** Colab Tesla T4 (`colab status`: `Hardware: T4`), device `cuda:0` (printed by Sections 3 and 5),
  torch 2.14.0+cu130, timm 1.0.29, isolated CPython 3.12.12 (45 locked packages, built in 55 s). **TF32:** the
  carried pipeline sets `torch.backends.cudnn.allow_tf32` and `torch.backends.cuda.matmul.allow_tf32` to `False` on
  CUDA (`pipeline.py`, amendment 1). The runtime record with `gpu_name` and the two TF32 flags goes only into
  `outputs/vit_classification_evaluation_report.json`, which stayed on the released VM. So TF32-off is confirmed
  from the carried source, not from printed output.
- **Score-once checks (Section 8).** `preregistered` True, `settings_differing` [], `grid_complete` True,
  `grid_problems` [], `state_matches_selected_system` True, `test_look` `pre-registered evaluation (this runtime)`.

**Validation grid** (validation log-loss by epoch; epoch 0 is the linear probe on frozen features):

| Config | Blocks / LR | Epoch 0 | 1 | 2 | 3 | 4 | Best epoch | Val. accuracy | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 2 / 3e-5 | 0.1621 | 0.1648 | 0.1655 | 0.1634 | 0.1660 | 0 (probe kept) | 0.9167 | 13.2 |
| C1 | 4 / 1e-4 | 0.1621 | 0.2370 | 0.1532 | 0.1282 | 0.1242 | 4 | 0.9583 | 15.8 |
| C2 | 2 / 3e-4 | 0.1621 | 0.2428 | 0.1375 | 0.1632 | 0.1673 | 2 | 0.9583 | 13.8 |

C1 is 0.0379 below C0 and C2 is 0.0246 below; both clear the 0.02 margin without lower validation accuracy, and the
lower score wins, so **C1 is selected** (28,351,488 block + 4,614 head parameters of 86,567,656; grid and selection
42.9 s). This **matches the masked CPU development check**: every config, epoch and summary line (train loss,
validation log-loss, accuracy, macro-F1; 21 lines) is identical at printed precision to the masked CPU runs of
`9919e13` and of round 2, with timings excluded. The selection-time validation look (+0.0417, CI [0, 0.125]) is
optimistic and decides nothing.

**Fresh test** (n = 96, 16 per species, 79 observers absent from the 180):

| System | Correct | Accuracy | Macro-F1 | Log-loss |
|---|---|---|---|---|
| Majority floor | 16 | 0.1667 | 0.0476 | — |
| Cosine 5-NN vote, frozen features (decides nothing) | 82 | 0.8542 | 0.8540 | — |
| Frozen probe | 80 | 0.8333 | 0.8315 | 0.4922 |
| Selected C1 (unfrozen last 4 blocks) | 78 | 0.8125 | 0.8115 | 0.5200 |

Primary endpoint (amendment 1): paired accuracy difference C1 − probe **−0.0208**, observer-cluster bootstrap 95 % CI
**[−0.0632, 0.0204]** (79 clusters, 10,000 resamples, seed 20261017); discordant photographs 1 (only C1 right) against
3 (only the probe right). Secondary, decides nothing: log-loss difference +0.0277, CI [−0.044, 0.1159]. **Verdict:
`no detectable difference`.** In plain words: **the adapted C1 did not beat the frozen probe.** It got two
photographs fewer right, and the interval includes 0. The 5-NN vote on the frozen features scored higher than
both; it decides nothing. This is one trained model (seed 0) against one probe, not a finding about unfreezing in
general.

**Exposed development test** (48 photographs; descriptives only, decides nothing): floor 0.1667 / macro-F1 0.0476,
5-NN 0.75 / 0.7516, probe 0.7917 / 0.7885 (log-loss 0.678), C1 0.7292 / 0.7182 (log-loss 0.791); C1 − probe −0.0625
accuracy, −0.0703 macro-F1, −3 photographs. These numbers equal the masked CPU check. Floor, 5-NN and probe equal the
2026-10-09 run. Predictions: 6 printed, 0 changed. The C1 adapter (50 tensors, 113,429,024 B) reloads to identical
probabilities, classes and `predict` output (exposed accuracy 0.729167 both ways). Section 10 was skipped
(`RUN_EXPERIMENT = False`).

**Re-derivation** (from the printed outputs; the bootstrap was not re-run because the per-photo predictions stayed
on the VM): 0.8125 × 96 = 78 and 0.8333 × 96 = 80, so the difference is −2/96 = −0.0208. The discordant counts give
the same net (1 − 3 = −2). The log-loss difference from the printed means is 0.0278, against 0.0277 printed (rounding
of unrounded means). The CI spans 0, so the rule gives `no detectable difference`, as printed. Fresh floor 16/96, 5-NN
82/96. Exposed: C1 35/48 (the confusion diagonal sums to 35), probe 38/48.

**Evidence** (`docs/execution-evidence/2026-10-18-1231036/`, `-text`, byte-for-byte):
`vit_classification_colab_1231036_colab-cli-t4_output.ipynb` sha256
`f43bd2c13cbfa20393e37027eb36381256f5eb0363f3db8995b9cdc5c6f6c801`; `run_summary.json` sha256
`ca464c18db397673d52153dfc4f71c9fb71ed6848d2ad805eb96ccfb324bb352`; `exec.log` redacted the same way as R108-S1: the
local run directory on lines 1 and 95 is replaced by `<run-dir>/` (2 occurrences, nothing else changed). SHA-256 as
captured `65a0c96e15a721200deeec21a1ba85668c580c52a6e4f9afe5b465dd644c86af`, as committed
`e982129a47daad548ae08fd5077ccb8f5f15704ff28a97aea214e3bfa31b4e0e`. Evidence boundary: CLI sequential execution, not
a browser Run all; forms and uploads not exercised; the status stays **Candidate** (REL12 BYOD and the Section 10
experiment pending).

## Primary ViT tutorial: hosted T4 reproduction 1 (2026-10-18)

**Label (REL13): reproduction of the pre-registered evaluation** (reproduction 1 of the `1231036` / `94e67934`
evaluation of record). Same pre-registered settings, run on a new fresh Colab T4 after the requote round; nothing was
changed in response to the test number.

- **Identity.** Commit `e96ee0e7a76ac5171dfb61ab7275a07fe5891c77`, notebook blob
  `bb9441e4542c4466a89a6472ac1cdfe8a334fa1c`; the run's `source.ipynb` hashes to that blob and is byte-equal to the
  committed notebook. `inspect_run.py`: 39/39 cells, same ids and order, source identity OK; embedded revision
  `f6b5b3a` (carries `RECORDED_EVALUATION`), generator `build_notebook.py/2.3`, `module_sha256` `63e9daad404e…`, spec
  2.5. 10 of 14 code cells have output. Session `suite-vit-e96ee0e-7e62`, `Hardware: T4`, device `cuda:0`, isolated
  CPython 3.12.12 (45 locked packages, 48 s).
- **Score-once checks (Section 8).** `preregistered` True, `settings_differing` [], `grid_complete` True,
  `grid_problems` [], `state_matches_selected_system` True, `test_look` `reproduction`; the fresh-test block also
  reads `reproduction`.
- **Match with the record.** `matches_record` (record `2026-10-18 1231036 / blob 94e67934`): `selected_config`,
  `selected_accuracy`, `probe_accuracy`, `primary_accuracy_difference`, `primary_accuracy_ci95`, `discordant_images`,
  `verdict` all True. A line-by-line diff of the two CLI logs (commit, revision and blob ids normalised) finds no
  result line that differs: grid curves and summaries for C0 / C1 / C2 (C1 selected, reason "C1 beat C0 by 0.0379
  validation log-loss"), fresh test C1 0.8125 (78/96) against the probe 0.8333 (80/96), macro-F1 0.8115 / 0.8315,
  log-loss 0.52 / 0.4922, difference −0.0208, CI [−0.0632, 0.0204], 79 clusters, discordant 1 vs 3, secondary
  log-loss difference and CI, floor and 5-NN references, `no detectable difference`; exposed development test
  descriptives and confusion; six predictions (0 changed); adapter 50 tensors, 113,429,024 B, same SHA-256 prefix
  `df3a2850e17f7d03`; reload parity identical (0.729167).
- **What differs** (expected): timings (session 262.5 s vs 317.0 s; environment 48 vs 55 s; corpus fetch 67.8 vs
  91.9 s; grid C0 / C1 / C2 12.5 / 15.1 / 13.2 vs 13.2 / 15.8 / 13.8 s; selection 40.8 vs 42.9 s), worker pid,
  embedded revision and `module_sha256`, `test_look`, the file suffixes, and the new `matches_record` line. The run
  also printed `matches_first` with every field False, because `first_report_readable` is False: the record's main
  report stayed on the record's released VM, so there was nothing on this VM to compare with. `matches_record` is the
  comparison that applies.
- **Files.** Section 8 and Section 9 wrote `vit_classification_evaluation_report_reproduction_1.json`,
  `…_predictions_reproduction_1.csv`, `…_result_reproduction_1.json` and `…_adapter_reproduction_1`; no main
  (unsuffixed) report, predictions, result or adapter was written. (`…_grid.json`, `…_input_manifest.json` and
  `…_train.csv` are written by the earlier sections under their usual names.)

**Evidence** (`docs/execution-evidence/2026-10-18-e96ee0e/`, `-text`, byte-for-byte):
`vit_classification_colab_e96ee0e_colab-cli-t4_output.ipynb` sha256
`417d78eec00f4452d22aa609f797722102cc262c76bbd865496982b4d2b4171b`; `run_summary.json` sha256
`62b7c3a8d15b39d4846d0bf08099eec5923014f01cda7b160a2afddcae235862`; `exec.log` redacted the same way as R108-S1: the
local run directory on lines 1 and 97 is replaced by `<run-dir>/` (2 occurrences, nothing else changed). SHA-256 as
captured `8840ee4dc60ca3cbd95263d4f61d2eda63534804c6aea10a690e1d326701e7b0`, as committed
`76d439cdf625609327cbbfa69b95948cd0976645e3574d514db42aec78ff64b9`. Evidence boundary: CLI sequential execution, not
a browser Run all; forms and uploads not exercised; the status stays **Candidate** (REL12 BYOD and the Section 10
experiment pending).

## Multi-model guided follow-up — 2026-09-26

Assessment uses the current fleet NOTEBOOK_SPEC 2.2 (Git blob `046d866eac7cc67b1539a4ba370ebc965341c87b`) and this repository's comparative notebook spec. The historic default STANDARD Colab run at notebook blob `c40bc9351916` remains valid for that artifact. This revision changes baseline evaluation timing and exported provenance, so it requires a new exact-revision hosted run. Its **Candidate** status does not alter the primary ViT tutorial's separate record above.

Confirmed gaps and dispositions:

| Gap | Fix / evidence |
|---|---|
| GDL5/7–9/11–12: objectives, predictions, stage interpretation and infrastructure separation incomplete | Observable objectives, architecture/metric explanations, pre-comparison predictions, worked checkpoints and collapsed carrier/runner cells added |
| GDL10: rerunning probe training after canonical freeze overwrites adapter files | Optional activity now requires two separate fresh runtimes, saved validation tables, only a step-budget change and a stop before freeze/test; cap selection no longer implies proven ongoing improvement |
| Early majority metrics expose test labels before freeze | Section 6 uses validation; fixed-rule test scoring moves to Section 14 after the frozen adapter evaluation |
| BYOD export claims six classes / 48 test images regardless of data | Actual class count and held-out image count are exported |
| BYOD prose implies grouped splits can be supplied | Documented that only automatic image-level stratification is implemented; external split folders/columns are not honored |
| REL12 and optional-tier execution evidence incomplete | Kept pending with concrete checks below; no release promotion |

Local baseline: release validator passed and 10/10 comparative notebook tests passed with `PYTHONPATH=src`. After: 18/18 targeted tests pass, release validator/generator parity pass, and changed Python files pass Ruff. New tests execute the notebook's actual BYOD directory/ZIP loading, image validation, split and manifest paths using 16 unique synthetic images across two classes; four invalid datasets are rejected (missing label, duplicate ID, missing image, too few per class). A sentinel proves the early baseline does not read test rows, then its fixed rule is scored against a different test distribution. The export-cell test checks actual BYOD counts using stand-in model results; it does not establish trained-model or reload correctness. No foundation models were loaded locally.

Reproduce in an environment with the repository's lightweight test dependencies:

```powershell
$env:PYTHONPATH='src'
python -m pytest --noconftest -o addopts= tests/test_multimodel_image_classification_workshop.py tests/test_workshop_learning_boundaries.py
python tools/validate_release_assets.py
python tools/build_multimodel_image_classification_workshop.py --check
```

`--noconftest` isolates these checks from model fixtures; this is not the full model-backed suite.

Remaining release evidence:

1. Run the exact revised notebook in a fresh supported Colab T4 runtime with default STANDARD settings. Retain commit/blob, runtime/package/device inventory, executed notebook, adapter reload results and report ZIP digest. Confirm validation-only majority output in Section 6 and test majority output after freeze in Section 14.
2. In another fresh runtime, set `USE_BYOD=True` and `BYOD_PATH` to a representative compatible directory or ZIP with `labels.csv` (`id,file,label`), at least eight independent images per class and 2–100 classes. Run the entire real-model path through feature extraction, training, selection, freeze, test, adapter reload and export. Record input/report digests and actual class/split counts. Synthetic parser fixtures are not this evidence.
3. In a separate negative run, remove the `label` column; require refusal at acquisition before models. Save the diagnostic and input digest. Do not count the intentional rejection as successful Run all.
4. Qualify FULL and the DINOv2 extension separately before claiming those paths verified. Exercise the optional two-runtime activity separately from canonical Run all; use only validation outcomes.

The remaining hosted and full-model BYOD gates prevent an unqualified gold-standard/release-grade claim for this revised comparative notebook.


## Maintainer-supplied Colab execution — 2026-09-26

The maintainer reported that this notebook passed an end-to-end Colab run and authorized merging its open PR. The supplied [executed notebook](execution-evidence/2026-09-26/DIMER_MultiModel_Image_Classification_Workshop.ipynb) is preserved byte-for-byte as evidence.

- Reviewed source commit: `1384256eb06be045a260cdf7f386bf75e946c1c5`.
- Executed-file SHA-256: `26c8195731af5493b50f5f6b7571a07056053a0c857660a30fd106e3bca4c698`.
- Independently inspected: 23 executed code cells; zero saved error outputs; terminal completion and exports present.
- Configuration/source comparison: STANDARD tier, DINOv2 extension disabled; only an added Colab title comment differs from reviewed code. Python ASTs match.
- Evidence boundary: saved outputs were inspected; execution was not independently repeated. This submission establishes the recorded path, not optional FULL/BYOD paths. Fresh-runtime/restart details beyond the maintainer's explicit prior confirmations are not inferred.

This record supersedes the pending rerun item for the source/configuration above. It does not promote the whole pipeline or close untested optional-path qualification.


## Multi-model workshop: Notebook Review Framework v1 findings — revision 0.2.0-candidate (2026-09-27)

A review under the Notebook Review Framework v1 examined commit `c92326b` (notebook blob `c50a1eb6`) and concluded **Needs revision**. It reported three major findings, three minor findings, one suggestion and one wording item. The review and its probes are archived in [`reviews/2026-09-27-notebook-review/`](reviews/2026-09-27-notebook-review/). Revision `0.2.0-candidate` (notebook blob `2cccb9c64fdb`) addresses every item.

`tests/test_image_workshop_review_fixes.py` executes the notebook's own cells end to end on a small BYOD fixture:

- acquisition, split, cache verification, the real probe-training and probe-inference programs (PyTorch and SafeTensors on CPU), freeze, test, galleries, export and the activity comparison;
- only the pretrained-backbone feature extractor is replaced, by a program with the same command-line and file contract that writes deterministic synthetic features.

These are contract checks, not backbone runs.

| Finding | Correction in 0.2.0 | Acceptance check |
|---|---|---|
| **ICR-01** (major): accepted BYOD labels broke at scoring (`0`, `001`, `NA`) or were truncated to 128 characters in the feature cache | Labels and IDs are text identifiers everywhere. `labels.csv` is read with `csv`; the feature cache stores `dtype=str`, which is not truncated; prediction files are read with `dtype=str` and no missing-value parsing. Ambiguous text (empty, leading or trailing space, control characters) is refused at acquisition. After extraction, each cache is checked against the validated dataset's IDs, labels and splits. Probe training requires every class in train and validation | Alphabetic, numeric, leading-zero, missing-token and 130-character Unicode label sets run through acquisition, cache, training, freeze, test and export unchanged; padded or control-character labels are refused before any cache exists |
| **ICR-02** (major): metrics were computed over whatever rows a prediction file held | `evaluate_prediction_csv` takes the authoritative split. It requires one row per expected image and no others, the dataset's label, columns in the frozen class order, finite probabilities within [0, 1] that sum to 1, and a prediction that is the highest-probability class. It reports `n_images`. The disagreement table starts from the full test grid | Missing, duplicate or extra rows, a changed truth, probabilities of 2.0, unnormalised or missing probabilities, a non-argmax prediction and reordered class columns are each rejected. A permuted file realigns by ID with identical scores |
| **ICR-03** (major): the freeze bound adapter tensors but not the manifest's class order, the feature cache or the runners | The freeze records each adapter's manifest digest and `class_order`, each feature-cache digest (the manifest also carries it), the dataset table, the three runner digests, and each environment's pins and signature. The test cell recomputes all of these and refuses on any difference. The inference program itself rechecks the feature, manifest and adapter digests before predicting. A freeze is never replaced, and the test runs once | A reversed `class_order`, changed feature values, changed split membership, adapter bytes, the inference runner, an environment signature, and a probe setting each stop scoring before any runner call. The inference program alone refuses a changed manifest. An unchanged re-freeze keeps the file; a second test is refused |
| **ICR-04** (minor): overlapping error-gallery captions | Each gallery column has the image with its rule and wrapped truth, then one model per line below it, marked ✓/✗. Gallery images are de-duplicated. The representative gallery adapts to the class count (up to 24, with a note) | Every caption line is at most 40 characters, including for 130-character labels; each model appears once per image; IDs are unique |
| **ICR-05** (minor): BYOD limits and privacy appeared late | Section 4 now carries the full contract table, label policy, privacy warning and a list of what the report exports, before the acquisition cell. Section 18 points back to it | Static order check |
| **ICR-06** (minor): directory-based export could include earlier runs' files | Each run of the controls cell creates `outputs/<experiment ID>/`. Runners, specs, the dataset table and BYOD extraction are per experiment. The ZIP holds only that folder, with `provenance/inventory.json` | A stale `test/eva.csv` in an earlier experiment stays out of the next report; the earlier report is byte-identical afterwards |
| **ICR-07** (suggestion): underspecified activity handoff | Section 12 writes a validation-only `activity/validation_record.json` with a digest. A new comparison cell loads two records, refuses tampered records or any difference other than `probe_steps`, and joins by model | Records hold no test result. A seed change is refused, a tampered record is refused, and a matched pair yields the per-model table |
| **ICR-08** (wording): selection narrative implied every probe found a minimum | Section 10 now says some probes were selected at the step cap in the reference run, and that a cap selection does not establish a minimum | Static wording check |

The generator source and the notebook had drifted since `c92326b`, because the AI-disclosure edit was made to the notebook only, so `test_generator_parity` failed on `main`. The disclosure now lives in `tools/multimodel_image_classification_workshop_source.py`, which restores parity. Three existing boundary tests were updated for the per-experiment output folder and the new export inputs.

Code cells changed, so no earlier hosted run describes this revision. **Status: Candidate.** The following exact-revision evidence is required:

- a fresh Colab T4 `STANDARD` run;
- BYOD with alphabetic and numeric-looking labels through export;
- a documented invalid-input rejection;
- the two-runtime activity using the saved records;
- separate `FULL` and DINOv2 qualification.

The review's learner-observation recommendation remains open.

## Maintainer-supplied Colab execution of revision 0.2.0 — 2026-09-28

The maintainer supplied an executed Colab copy of revision `0.2.0-candidate`. It is preserved byte-for-byte as [evidence](execution-evidence/2026-09-28/DIMER_MultiModel_Image_Classification_Workshop.ipynb).

- Reviewed source: commit `1ef3d0f`, notebook blob `2cccb9c64fdb`. All 56 cell ids and sources match the committed notebook exactly.
- Executed-file SHA-256: `53bd2c326db500559cafd80fb1b69534b5985e9919d5c34c4d72452710610bf7`.
- Runtime: Google Colab, Tesla T4; host Python 3.13.15, NumPy 2.1.3, pandas 2.2.3, Matplotlib 3.10.0, Pillow 11.3.0.
- Execution: 24/24 code cells ran in order, with no saved error outputs. The terminal completion summary and the report export are present.
- Results: identical to the recorded `e9237d0` run to the printed precision (table above). The new checks behaved as intended on this path. Every model was scored over exactly the 48 test images. The disagreement table covers the full test grid. The export holds only `outputs/20260928T020445Z-6f8082/` (50 files, bundle SHA-256 `f8aa524b5c50dc23dbc981cbb7369ee81a957d47d946b9923339dd257466ea73`).
- Evidence boundary: saved outputs were inspected; the execution was not independently repeated.

This closes the "fresh Colab T4 `STANDARD` run" item above. The following remain open:

- BYOD with alphabetic and numeric-looking labels through export on a hosted runtime (covered by the notebook-cell tests only);
- a documented invalid-input rejection on a hosted runtime;
- the two-runtime activity using the saved records;
- `FULL` and DINOv2 qualification;
- the learner-observation recommendation.

**Status: Candidate.**

## Multi-model workshop: 2026-10-03 uv isolated environment — revision 0.3.0-candidate

Notebook blob `2cccb9c64fdb` (revision `0.2.0-candidate`, base `c3f5030`) → `86f36b4733cc` (revision `0.3.0-candidate`). Code cells changed, so the 2026-09-28 Colab run above does not describe this blob. **A hosted re-run is pending. Status: Candidate.**

What changed:

- Section 7 builds each model environment from a hash lock. It verifies a pinned `uv` 0.12.15 wheel by size and SHA-256 and runs `uv venv --managed-python --python 3.12.12`. Then it runs `uv pip install --require-hashes --only-binary :all:`. Before, it ran `pip install uv` into the kernel when `uv` was missing and installed unhashed direct pins with the environment's `pip`. Nothing is installed into the kernel any more, and Run all needs no restart.
- The two locks (`tools/image-workshop-common-requirements.lock`, `tools/image-workshop-swin-requirements.lock`) are compiled by `uv pip compile --generate-hashes` for CPython 3.12 on `x86_64-manylinux_2_28`. They are carried byte-for-byte in a new carrier cell (`uvcarrier`) and verified by SHA-256. The direct pins are unchanged.
- Model processes no longer inherit `PYTHONPATH`, `PYTHONHOME` or `PYTHONSTARTUP` from the kernel. `MPLBACKEND=Agg` is kept.
- The embedded photo manifest is a parenthesised run of 1,000-character pieces. Its `ast.literal_eval` value is asserted equal to the source literal. No notebook line exceeds 2,000 characters; the longest was 16,211.
- **User-visible:** the notebook now runs on **Linux x86_64 only** (Google Colab, Kaggle, Linux Jupyter). Section 7 refuses Windows and macOS kernels.

The ICR-01 to ICR-08 fixes are unchanged, and their tests pass. `tests/test_image_workshop_uv.py` covers the new contract.

## Colab CLI execution of revision 0.3.0 — 2026-10-03

Notebook blob `86f36b4733cc` (revision `0.3.0-candidate`, commit `e8d1dea`) was executed on a fresh Colab Tesla T4 by the Colab CLI. The executed notebook is preserved byte-for-byte as [evidence](execution-evidence/2026-10-03/DIMER_MultiModel_Image_Classification_Workshop_e8d1dea_colab-cli-t4.ipynb).

- Source: fetched from GitHub at the full commit SHA `e8d1dea58015005579313d24edd8c23c5e4637bb`; its git blob `86f36b4733ccf24b8ff9d4ff8ac1fb65f4f0d89f` was checked before the VM was allocated. All 57 cell ids and sources of the executed copy match the committed notebook.
- Executed-file SHA-256: `f77b32e5dcaaceb3f06e454bec82b0c94e005c1564167f669ea141b7a24bc881`.
- Executor: Colab CLI 0.7.4, fresh Colab Tesla T4 (session released afterwards). Host Python 3.13.15, NumPy 2.1.3, pandas 2.2.3, Matplotlib 3.10.0, Pillow 11.3.0. Both carried hash locks verified; the common model environment was built as CPython 3.12.12 by the pinned `uv`, with no kernel install and no restart.
- Execution: 25/25 code cells ran in order in one kernel (CLI log `Executing cell 1..25/25`), with no error outputs; 204.6 s for the whole session.
- Results: every quality metric equals the 2026-09-28 Colab run of blob `2cccb9c64fdb` (split, dataset digest, observer overlap, selected steps, validation and test accuracy / macro-F1 / log-loss, majority floor, disagreement counts, reload parity). The output diff (difflib on normalised lines) shows only:
  - `feature_seconds` and `images_per_second`: wall-clock timings, which vary between runs;
  - two new log lines from the uv environment build (`Carried hash locks verified`, `Creating the common environment`): new code in this revision;
  - `notebook_revision` `0.2.0-candidate` → `0.3.0-candidate`, and the report bundle SHA-256 (`f8aa524b5c50…` → `56d0b7cf951f…`): the bundle carries the experiment id, the timings and the revision string.
- Evidence boundary: Colab CLI sequential execution, not a browser Run all; the default `STANDARD` path only. Forms were not rendered, and BYOD, the activity comparison, `FULL` and DINOv2 were not exercised.

This closes the hosted re-run item for revision `0.3.0-candidate`. The items listed open under the 2026-09-28 record remain open.

**Status: Candidate.**

## Multi-model workshop: ICR2 review fixes — revision 0.4.0-candidate (2026-10-15)

Notebook blob `86f36b4733cc` (revision `0.3.0-candidate`, base `6c7b057`) → revision `0.4.0-candidate`, from the W21 capstone-round review (findings ICR2-M1, ICR2-m1 to m6, ICR2-S1), assessed against NOTEBOOK_SPEC **2.5** (ml-worker `bfe4150`). Code cells changed, so the 2026-10-03 Colab CLI run above does not describe this blob. **A hosted re-run is pending. Status: Candidate.**

Owner decision ICR2-M1 = C (Kurt): disclose each checkpoint's pretraining label sets **and** add ImageNet-1k-only counterparts for a comparison with matched supervised label vocabulary.

What changed:

- **Pretraining labels (ICR2-M1).** Section 2 tabulates, per checkpoint, its supervised label sets and which target species are classes in them, from the synset lists in `timm/data/_info`: all six species are ImageNet-12k/21k/22k classes; ImageNet-1k has house finch exactly and a broader "junco, snowbird" class (n01534433). The title, opening cell, Section 15, Section 21, exercises A to D and the conclusion template no longer credit the accuracy gap to architecture or size.
- **ImageNet-1k-only counterparts (ICR2-M1, option C).** `RUN_IN1K_COUNTERPARTS = True` (default) adds `timm/convnext_tiny.fb_in1k` at `b43a6303c9fc…` and `timm/vit_base_patch16_224.augreg_in1k` at `458542882691…` (Apache-2.0; pinned by size and SHA-256 like the core models), run with the same probe recipe after the core models. Section 14 prints a table of the ImageNet-1k-only checkpoints and each core checkpoint against its counterpart. The core rows and their computation are unchanged; the counterparts match architecture and size but not the whole pretraining recipe, which the notebook states.
- **Runner.** The feature runner now always sets `custom_load=False` in the pretrained-config overlay. The ViT `augreg_in1k` config sets `custom_load: true`, which makes timm 1.0.29 read the pinned safetensors file as an original JAX `.npz` and fail; all previously pinned configs already had `custom_load: false`, so their loading is unchanged (checked on CPU: the loaded weights equal the file for both counterparts).
- **DAT9 (ICR2-m1).** Sections 0 and 21 state that overlap between the public iNaturalist photos and the pretraining images cannot be ruled out.
- **KND10 (ICR2-m2).** Sections 0, 10 and 21 say the probe recipe is fixed and not tuned per model, and that results hold under this recipe only.
- **Paired evidence (ICR2-m3).** Section 14 prints nominal Wilson 95% intervals (independent photos assumed); Section 15 prints paired discordant counts with an exact McNemar p for every pair of tested checkpoints (illustrative, no multiple-comparison correction, written to `test/paired_comparisons.csv`). The Section 21 interval sentence is reworded.
- **BYOD refusals (ICR2-m4, DAT20).** A contract problem is refused with one message listing every problem found (file or `labels.csv` line and rule) and a recovery step, with no traceback; Run all stops there. Undecodable images still stop with their own error.
- **REL13 (ICR2-m5).** The learning-rate-0.01 result is labelled as recorded at `b9b94c9` / blob `5874024a` and not re-run; the step-cap statement names the 2026-10-03 run of `e8d1dea` / `86f36b47`.
- **Declaration (ICR2-m6).** `metadata.dimer.notebook_spec` 2.5, `notebook_kind` `WORKSHOP`; the opening cell, provenance and completion summary say the same.
- **Majority tie (ICR2-S1).** Section 6 prints a notice when the majority label is a tie-break.

Expected unchanged on a hosted run: the 108 / 24 / 48 split, the dataset digest and every core-model metric of the 2026-10-03 run (inferred from the diff: each model runs in its own process with the same inputs and seed; not re-executed). New numbers (the counterparts, intervals and paired tables) come only from a hosted run of this blob.

Verification (not clean-runtime evidence): generator `--check`, release-asset validation and ruff pass; the review's 7 acceptance checks fail before and pass after; new tests in `tests/test_multimodel_workshop_icr2_fixes.py` run the notebook's own cells with stand-in predictions and synthetic BYOD folders.

Open: a hosted default run of this blob; REL12 hosted BYOD accept and refuse runs (the refusal rendering on Colab is not yet observed); the two-runtime activity; `FULL` and DINOv2.

**Status: Candidate.**

### Follow-up after the peer review of PR #13 (still revision 0.4.0-candidate)

A peer review of the first ICR2 commit found the scientific framing still overstated in places. Changes, all wording, displays or additive exports; no core computation changed:

- The ImageNet-1k-only group is now a "comparison with matched supervised label vocabulary". The two core-versus-counterpart pairs (ConvNeXt-Tiny 28.59 M parameters and ViT-B/16 86.57 M in each pair, with identical weight-file sizes) are the planned teaching contrasts. The notebook says that neither the group nor the pairs isolates architecture or label coverage, and prints each pair's resolved preprocessing.
- Exact-label wording: all six exact species labels occur in the larger label sets; ImageNet-1k contains house finch exactly and a broader junco class, and the other four exact labels are absent.
- "Independent test" became "held-out tutorial split". The notebook notes that reuse across revisions makes comparisons exploratory.
- McNemar p-values are labelled nominal and exploratory. The notebook explains observer dependence, balanced 8-per-species sampling and multiplicity. The planned contrasts keep their nominal p; other pairs show a Holm-adjusted p and stay descriptive. The conclusion template separates "more correct on this sample" from evidence of higher accuracy in general.
- Validation scores are described as optimistic selection results, and the seed as a reproducibility control. Section 11 prints both boundary notes (first step, step cap) without a causal diagnosis.
- Throughput is labelled as observed end-to-end extraction throughput at the listed batch size, with device. The accuracy-per-million-parameters ratio was removed.
- One `NOTEBOOK_SPEC` constant (2.5) feeds the frozen identity, which still said 2.1, the provenance and the completion summary. The provenance cell checks that they agree.
- `test/aggregate_metrics.csv` and the computational trade-off table cover every tested checkpoint. Confusion matrices include the counterparts.
- Simple reference instead of "floor", plus the uniform-probability log-loss ln(6) ≈ 1.792; "checks exact duplicate content"; per-species denominators (1 photo = 12.5 points of a species' recall).

**Status: Candidate.**

## Colab CLI execution of revision 0.4.0 — 2026-10-15

Notebook blob `be5203b841ea` (revision `0.4.0-candidate`, commit `ed4b542`) was executed on a fresh Colab Tesla T4 by the Colab CLI. The executed notebook is preserved byte-for-byte as [evidence](execution-evidence/2026-10-15/DIMER_MultiModel_Image_Classification_Workshop_ed4b542_colab-cli-t4.ipynb).

- Source: fetched from GitHub at the full commit SHA `ed4b54297b67bb1c1e66cf9b079b3a512a8fecf6`; its git blob `be5203b841ea6a208c546a413f1051165a04ea43` was checked before the VM was allocated. All 60 cell ids and sources of the executed copy match the committed notebook.
- Executed-file SHA-256: `6ac6bef4c4f70964db0316f26631f82f55a7607de4c383dd81109f0b880595ff`.
- Executor: Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-vit-ed4b542-4df8`, released afterwards), 2026-10-15 01:56–02:00 UTC. Host Python 3.13.15, NumPy 2.1.3, pandas 2.2.3, Matplotlib 3.10.0, Pillow 11.3.0. Both carried hash locks verified; the common model environment was built as CPython 3.12.12 by the pinned `uv`, with no kernel install and no restart.
- Execution: 27/27 code cells ran in order in one kernel (CLI log `Executing cell 1..27/27`), with no error outputs; 234.0 s for the whole session. The provenance cell's spec-consistency check (frozen identity, manifest and constant all `2.5`) passed; the completion summary reports `notebook_spec` 2.5.
- ImageNet-1k-only counterparts, first hosted run: `timm/convnext_tiny.fb_in1k` at `b43a6303c9fc…` and `timm/vit_base_patch16_224.augreg_in1k` at `458542882691…` were downloaded, size- and SHA-256-verified by the feature runner (which raises on a mismatch), and loaded through the `custom_load=False` state-dict path; both extracted features on `cuda:0` (768-d) and trained probes (selected at steps 40 and 50, validation accuracy 0.833 and 0.708, reload parity 0.0). They appear in `test/aggregate_metrics.csv` (the test table), the six confusion matrices and the trade-off table.
  - ConvNeXt-Tiny (ImageNet-1k only): 27/48 = 0.562, nominal Wilson 95% [0.423, 0.693], macro-F1 0.556, log-loss 1.127.
  - ViT-B/16 (ImageNet-1k only): 22/48 = 0.458, nominal Wilson 95% [0.326, 0.597], macro-F1 0.459, log-loss 1.481.
  - Planned contrasts, nominal exact McNemar: ConvNeXt-Tiny (ImageNet-12k → 1k) vs its counterpart 34 vs 27 correct, discordant 8 / 1, p = 0.0391; ViT-B/16 (ImageNet-21k → 1k) vs its counterpart 37 vs 22, discordant 16 / 1, p = 0.0003. The other 13 pairs are descriptive with Holm-adjusted p. The Wilson bounds and both p-values were recomputed independently from the printed counts and agree.
- Results vs the 2026-10-03 run of blob `86f36b4733cc`: every core-model quality metric is identical (split, dataset digest, observer overlap, selected steps, validation and test accuracy / macro-F1 / log-loss, majority floor, disagreement counts over the core models, reload parity), as predicted in the 0.4.0 record above. The output diff (difflib on normalised lines) shows only:
  - the two counterparts' rows, extraction and probe lines, the label-set disclosure table, the Wilson / matched-vocabulary / planned-contrast / preprocessing tables, the all-pairs McNemar table, and two more confusion matrices: new in this revision (ICR2-M1 option C, ICR2-m3);
  - `feature_seconds` and `images_per_second`: wall-clock timings; the throughput table now also prints batch size and device, and the accuracy-per-million-parameters column is gone (peer-review follow-up);
  - the feature-runner SHA-256 (`d94e26e8…` → `79dc6b8a…`): the `custom_load=False` overlay;
  - the majority-tie notice and the uniform-probability log-loss (ICR2-S1, follow-up);
  - `notebook_revision` `0.3.0` → `0.4.0-candidate`, `notebook_spec` 2.1 → 2.5 (ICR2-m6), the experiment id, and the report bundle (50 → 67 files, SHA-256 `56d0b7cf951f…` → `9fc4c6b56dff…`): more models and the new exports.
- Evidence boundary: Colab CLI sequential execution, not a browser Run all; the default path only. Forms were not rendered, and BYOD (including the ICR2-m4 refusal rendering), the activity comparison, `FULL` and DINOv2 were not exercised.

This closes the hosted re-run item for revision `0.4.0-candidate`. The other open items above remain open.

**Status: Candidate.**

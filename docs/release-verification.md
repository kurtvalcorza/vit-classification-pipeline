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
  adaptation to a label space the checkpoint does not have under an explicit frozen-vs-unfrozen policy, the majority
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
   `LEARNING_RATE = 3e-5`, `BATCH_SIZE = 8`, `TRAINABLE_BLOCKS = 2`);
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

Notebook blob `c40bc9351916` (from `e9237d0`) has a passing Google Colab T4 run of the default `STANDARD` path. The
`FULL` tier (SwinV2-Tiny, EVA-02 Base 448) and the optional DINOv2 baseline have not been exercised on a hosted
runtime. As for the other notebooks, this table is the evidence record: `metadata.dimer.clean_runtime_evidence`
stays `pending` as authored, because editing it would change the blob these runs verify, and the registry status
stays **Candidate** until a reviewer promotes it.

## Current status

**Release-grade.** The `E2E` notebook blob `50462c07` (committed at `8b26ec6`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 276.0 s, 188 files, 365 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

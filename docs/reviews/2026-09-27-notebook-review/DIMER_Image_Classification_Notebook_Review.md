# DIMER Multi-Model Image Classification Notebook Review

**Verdict: Needs revision**  
**Date:** 27 September 2026  
**Framework:** Notebook Review Framework — v1, supplied in this conversation  
**Notebook:** `tutorials/DIMER_MultiModel_Image_Classification_Workshop.ipynb`  
**Repository:** `kurtvalcorza/vit-classification-pipeline`  
**Reviewed commit:** `c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7`  
**Notebook Git blob:** `c50a1eb660759c958b3f0cf20b66d86fae987ea9`  
**Declared profile / mode / specification:** `E2E` / `WORKSHOP` / `2.1`  
**Additional conformance baseline inspected:** NOTEBOOK_SPEC 2.2, dated 2026-09-26, blob `7428d5becb8d37133a3ad93a450d08ebf616f411`.

## 1. Executive assessment

The central teaching design is coherent: compare available pretrained vision systems on the same image-classification task using frozen feature extraction, training-only feature normalization, a common linear-probe recipe, validation selection, and a held-out test. The notebook explicitly acknowledges that pretraining and native transforms differ; it does not claim to isolate architecture alone. That distinction should be retained.

Three major findings need correction: accepted BYOD labels do not reliably survive serialization, prediction evaluation does not verify the complete declared evaluation set, and the freeze does not bind all files that determine inference semantics. A separate rendered-layout defect makes the central error gallery hard to read. These do **not** establish that the recorded default bird-classification run produced incorrect metrics.

Keep the instructional structure. Repair the data/evaluation/artifact boundaries and the gallery, then qualify the changed paths. Adding more introductory text would not resolve the demonstrated technical discrepancies.

### Separate judgments

| Area | Judgment |
|---|---|
| Promise fulfillment | Substantial for the built-in comparison; incomplete for the general BYOD promise and the integrity implied by frozen, like-for-like evaluation. |
| Technical correctness | Needs revision at class-label round trips, prediction validation, and semantic artifact binding. The synthetic-feature probe loop and reload positive control worked locally. |
| Scientific/experimental validity | The stated comparison is appropriately limited to pretrained systems. Train-only normalization is correct in the inspected source. Unchecked evaluation coverage/labels and mutable inference inputs weaken the evidence boundary. |
| Learner experience | Good task framing and interpretation prompts; error-gallery text overlaps, and the two-runtime activity needs a more explicit save/handoff mechanism. |
| Specification conformance | Targeted gaps identified; not a complete certification against all requirements. The notebook declares 2.1; inspected 2.2 requirements are identified separately below. |
| Execution readiness | Relevant STANDARD execution is documented by the repository. This review did not independently repeat or re-inspect the complete archived Colab execution. FULL, DINOv2, full-model BYOD and the changed implementation need their own evidence. |

## 2. Review contract and evidence

### Intended learner and promised outcomes

The intended reader can open/run Python cells in Colab or Jupyter but is new to transfer learning and modern vision architectures. The canonical runtime is a CUDA GPU such as a Tesla T4. STANDARD selects MobileNetV4-Conv-Small, ResNet-50, ConvNeXt-Tiny and ViT-B/16; FULL adds SwinV2-Tiny and EVA-02 Base 448. DINOv2 is a separately labelled optional representation baseline.

The stated built-in dataset contains 180 bird photographs from six species, split into 108 training, 24 validation and 48 test images. The notebook says it verifies each download against its embedded manifest and detects identical decoded pixels across splits. This review inspected those mechanisms but **did not download or independently validate the actual 180 photographs**.

The learner is expected to explain frozen features versus a trained probe, compare against a majority baseline, interpret accuracy and log-loss, inspect class-specific errors/disagreements, consider computational cost, and write a bounded conclusion.

Source: [opening and notebook](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tutorials/DIMER_MultiModel_Image_Classification_Workshop.ipynb); [source-cell definitions](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tools/multimodel_image_classification_workshop_source.py).

### Evidence actually used

**Source inspection:** notebook and source-cell definitions, the repository's release record and learning-boundary tests, and relevant fleet-specification sections. The notebook source contains the functions and calls discussed here; the source-cell module was used to inspect long embedded runner strings without the notebook JSON escaping.

**Documented execution:** the latest relevant release-record entry reports a maintainer-supplied STANDARD Colab run, DINOv2 disabled, for reviewed source commit `1384256eb06be045a260cdf7f386bf75e946c1c5`. It records 23 executed code cells, no saved error outputs, completion/exports, and AST agreement except for a Colab title comment. The retained executed-file digest is `26c8195731af5493b50f5f6b7571a07056053a0c857660a30fd106e3bca4c698`. This is evidence reported by the repository, not a run performed in this review. Earlier records include the probe-learning-rate correction and associated historical STANDARD metrics. The primary single-model ViT tutorial has a separate status and must not be conflated with this multi-model notebook.

**Local execution:** 13 grouped checks on transcribed source excerpts and synthetic fixtures. They include real bounded PyTorch linear-probe fitting, real SafeTensors serialization/reload, and a separate-process invocation of the transcribed probe-inference runner. No pretrained vision backbone or native image transform was executed. All resulting test scores are synthetic-fixture results, not bird-classification measurements.

**Visual inspection:** the error-gallery layout was reproduced with plain image blocks and realistic built-in class-name strings; the resulting PNG was inspected. It is not a screenshot of the actual Colab run.

**Unavailable/unverified:** fresh Colab/T4 execution; live downloads and environment installation; full-model feature extraction; all-model feature-cache identity comparison on the real corpus; complete model-backed BYOD; FULL/DINOv2 execution; downstream serving compatibility; representative learner observation. Container network retrieval was unavailable, so the probe package uses clearly labelled transcribed excerpts rather than a byte-for-byte notebook executor.

Local environment: Python 3.13.5, Linux, NumPy 2.3.5, pandas 2.2.3, Pillow 12.3.0, matplotlib 3.10.8, PyTorch 2.10.0+cpu, SafeTensors 0.7.0. These are **not** the notebook's declared model-environment pins.

Execution source: [release-verification.md, latest entry](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/docs/release-verification.md#maintainer-supplied-colab-execution--2026-09-26). Existing acquisition-oriented tests: [test_workshop_learning_boundaries.py](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tests/test_workshop_learning_boundaries.py).

## 3. Major findings

### ICR-01 — Accepted BYOD class labels do not survive the complete workflow

**Severity:** Major.  
**Locations:** Section 4 `load_byod`; Section 8 feature export; Section 12 `evaluate_prediction_csv`, reused in Section 14.  
**Dimensions:** promise fulfillment, technical correctness, interaction/recovery, completion/transfer.

`load_byod` reads labels with `csv.DictReader`, preserving literal strings. The common evaluator later executes `pd.read_csv(path)` without specifying label dtypes or missing-value parsing. It indexes `class_names` using the interpreted `truth` values. Numeric-looking labels become integers; a literal `NA` can become missing. The lookup then fails even though acquisition accepted the labels.

#### Observed reproductions

Each case used 16 unique synthetic PNGs for acquisition and an independently constructed synthetic feature cache for the common probe. No visual representations were computed.

| Accepted label set | Reloaded truth dtype | Scoring result |
|---|---|---|
| `alpha`, `beta` | text | Positive control succeeds. |
| `0`, `1` | `int64` | `KeyError: 0`. |
| `001`, `002` | `int64` | `KeyError: 1`; leading zeros are lost. |
| `NA`, `other` | object with a missing value | `KeyError: nan`. |

A second loss-of-identity path exists in feature export: labels are explicitly stored as `np.asarray(labels, dtype="U128")`. The acquisition contract has no corresponding length limit. Two accepted 129-character labels that differ only in their final character became the same 128-character cached label. This can collapse class identity before training and subsequently cause a mismatch with the parent class vocabulary.

**Learner consequence:** valid-looking custom data can get through image checks and expensive feature extraction before failing at the metric stage. A class label is an identifier, not a number to reinterpret or text to truncate silently.

**Recommended correction:** preserve literal class labels throughout CSV and NPZ boundaries. Apply explicit text/missing-value policies to `id`, `truth` and `prediction`; validate probability columns numerically. Store complete labels, or reject a clearly documented limit before model acquisition. Check class-vocabulary equality and per-class coverage after feature reload. Do not use object/pickle arrays merely to avoid the fixed-width issue.

**Acceptance check:** carry alphabetic, numeric, leading-zero, literal missing-token and long Unicode labels through acquisition, cache creation, training, reload, evaluation and export. Ambiguous or unsupported input must be rejected early and specifically, never silently remapped.

**Relevant 2.2 requirements:** DAT13–DAT14, DAT19, VAL2/VAL5, VAL7. These are targeted mappings, not a claim that every listed clause fails independently.

Sources: [acquisition source, lines 56–70](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tools/multimodel_image_classification_workshop_source.py#L56-L70); [feature runner, lines 115–123](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tools/multimodel_image_classification_workshop_source.py#L115-L123); [evaluator, lines 154–160](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tools/multimodel_image_classification_workshop_source.py#L154-L160). Library semantics are documented in [pandas read_csv](https://pandas.pydata.org/docs/dev/reference/api/pandas.read_csv.html) and [NumPy fixed-width string documentation](https://numpy.org/doc/2.0/user/basics.strings.html). The observation above is from the local versions recorded here, not from executing those documentation versions.

### ICR-02 — Metrics do not prove that models were scored on the same complete test set

**Severity:** Major.  
**Locations:** Section 12 `evaluate_prediction_csv`; Section 14 test scoring; Section 15 disagreement merge.  
**Dimensions:** scientific validity, technical correctness, promise fulfillment, interpretation.

The evaluator trusts `truth` and `prediction` in the prediction CSV and computes metrics over whatever rows are present. It receives no authoritative expected split. It does not check exact image-ID coverage, duplicate IDs, agreement with dataset labels, probability normalization/range, or agreement between the predicted label and probability argmax.

The disagreement table performs inner joins across model outputs. Missing predictions can therefore disappear from the comparison. Its `validate="one_to_one"` detects duplicate merge keys at that later stage, but does not retroactively validate already displayed metrics or establish complete image coverage.

#### Observed reproductions

A synthetic six-class fixture contained 48 predictions with one deliberate error:

| Input to evaluator | Result |
|---|---|
| All 48 predictions | Accuracy 47/48 = 97.9167%. |
| The erroneous row omitted | Accepted; accuracy 100% on 47 rows. |
| The erroneous row replaced by a duplicate correct-ID row | Accepted at the metric stage; accuracy 100% on 48 rows. The later disagreement merge may reject duplicates. |
| The erroneous row's `truth` changed to match its predicted label | Accepted; accuracy 100%. |
| Every probability set to 2.0, with chosen labels equal to truth | Accepted; accuracy 100%, clipped log-loss 0.0, despite invalid distributions. |

These are deliberate fault injections, not evidence that the archived model run produced any of these files.

**Learner consequence:** a metric table can appear to compare models fairly while rows or labels differ. Missing difficult images can improve the score. An invalid score vector can look like a highly confident correct prediction rather than a contract violation.

**Recommended correction:** make the evaluator accept the expected split manifest and class order. Require exactly one prediction for each expected image ID, no extra IDs, and target equality with the authoritative labels. Check finite probabilities within tolerance, row sums, class order, and argmax consistency before metric calculation. Align the disagreement table to that validated expected grid and report the denominator.

**Acceptance check:** complete correct files pass; missing/extra/duplicate IDs, changed truth, malformed probability vectors, and inconsistent class decisions fail before any successful metric table/export. A harmless row permutation should realign by ID and retain scores.

**Relevant 2.2 requirements:** EVAL2/EVAL8, OUT4; supports the notebook's explicit common-data/evaluation claim. The exact-grid checks are proposed implementation controls, not a verbatim list from those requirements.

Source: [evaluator](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tools/multimodel_image_classification_workshop_source.py#L154-L160); [test scoring and disagreement cells](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tools/multimodel_image_classification_workshop_source.py#L168-L199).

### ICR-03 — The freeze binds adapter tensors but not the complete inference state

**Severity:** Major.  
**Locations:** Sections 10, 13 and 14; `PROBE_INFER_RUNNER`, `artifact_digest`, frozen-test assertions.  
**Dimensions:** technical correctness, experimental validity, promise fulfillment, completion/transfer.

The test cell checks the dataset-digest variable, model set, checkpoint revision and the hash of `adapter.safetensors`. Inference also consumes the adapter's `manifest.json`, including `class_order`, and the cached `.npz` features/labels/split IDs. Neither of those files is bound to the freeze by a verified content digest. The runner/source identity is also not part of this check.

The class order is not decorative metadata: it determines how output columns become labels. Matching tensor bytes alone therefore does not establish equivalent classification behavior.

#### Observed reproductions

Using a genuinely trained two-class synthetic probe and the transcribed inference runner in separate Python processes:

1. The original adapter correctly classified all four synthetic held-out feature rows.
2. Reversing only `manifest.json`'s `class_order` left every transcribed freeze assertion passing and changed all four decoded predictions. Adapter bytes were unchanged.
3. Restoring the manifest and changing only the cached test feature vectors also left the freeze assertions passing and changed all four predictions.
4. Modifying the adapter bytes themselves was rejected, confirming that the implemented tensor check is functioning.

These are controlled integrity probes, not evidence of accidental mutation in a historical run. The notebook already warns that changing controls requires a fresh runtime; that useful instruction does not bind all semantic files in the current experiment.

**Recommended correction:** include and verify the complete adapter manifest, class order, feature-cache digest and row/split identities, plus material runner/configuration identity. Establish the cache's relationship to the validated dataset and model/native transform. Check before inference, not only before export. Keep an explicit new-experiment boundary rather than silently replacing a completed freeze.

**Acceptance check:** changes to class order, cached feature values, IDs/labels/split membership, adapter tensors or relevant runner code must fail before scoring, or initiate a clearly distinct experiment. Unchanged files must preserve reload parity.

**Relevant 2.2 requirements:** ART2/ART5, OUT8, VER2–VER4, EVAL8. This is a semantic-integrity gap; it is not a claim of executable deserialization of arbitrary Python via SafeTensors.

Source: [probe training/inference runners](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tools/multimodel_image_classification_workshop_source.py#L132-L143); [freeze and test](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tools/multimodel_image_classification_workshop_source.py#L168-L179).

## 4. Learner-facing and secondary findings

### ICR-04 — Error-gallery captions overlap

**Severity:** Minor; fix before workshop use because it directly obstructs the disagreement activity.  
**Location:** Section 15, deterministic error gallery.

The code puts four model/label pairs on one unwrapped line above each image in a three-panel figure. With realistic built-in class-name strings, a local rendering produced title widths of approximately 709, 817 and 592 pixels over axes around 443 pixels wide. Both adjacent pairs of titles overlapped. The exported PNG visibly contains overwritten text.

The diagnostic table remains available, so this is not classified as a missing entire diagnostic capability. The qualitative gallery itself is difficult to use as instructed.

**Correction:** use one model per caption line, wrap within a bounded panel, or put predictions in a table beneath/alongside each image. Deduplicate selected gallery IDs when multiple rules choose the same image. Adapt the introductory fixed 2×3 gallery for BYOD class counts rather than silently showing at most six classes.

**Acceptance:** visually inspect the in-notebook and exported layouts for STANDARD, FULL and long BYOD labels; text must remain readable without shrinking the whole figure to accommodate overlong titles. The included synthetic PNG demonstrates the current layout issue, not actual model predictions.

### ICR-05 — Input limits and privacy context appear too late

**Severity:** Minor; with a targeted conformance issue under 2.2 DAT12/VAL6.

The first BYOD instructions state `id,file,label` and directory/ZIP support. The 2–100-class rule, 4096-pixel side ceiling, 2 GiB expanded ZIP limit and eight-images-per-class minimum are scattered between collapsed code and later troubleshooting. The substantive privacy/group-independence discussion appears in Section 18, after the acquisition/training path.

**Correction:** move a compact BYOD contract immediately before the input-selection/upload cell, including bounds, supported image forms, label handling, automatic image-level splitting, and the hosted-runtime privacy warning. Describe what the report exports, including representations, labels/metadata and example images in figures. This does not require adding a new workflow.

**Acceptance:** a learner should be able to decide whether their data are compatible and authorized for the runtime without reading the implementation or reaching Section 18 after upload.

### ICR-06 — Export membership is directory-based, not current-run based

**Severity:** Minor hardening item, not an established defect in the documented two-fresh-runtime activity.

`shutil.make_archive(..., root_dir=OUTPUT_DIR)` includes the entire directory. A local reuse fixture retained an old FULL-tier `eva.csv` in a later STANDARD report. However, the notebook explicitly requires separate fresh runtimes for changed controls and warns that changing only `OUTPUT_DIR` does not isolate the shared work directory. Following those instructions avoids this particular reuse case.

**Correction:** use a current-run allowlist or experiment-specific result directory, with reusable weights/images outside it. Avoid blindly deleting arbitrary user-selected output paths.

**Acceptance:** pre-existing unrelated output and prior-tier files do not enter a new report; old reports remain intact. Treat accidental-state exclusion separately from whether the report intentionally includes reusable adapters or audit features.

### ICR-07 — The active-learning handoff is underspecified

**Severity:** Suggestion.

The step-budget activity correctly specifies two fresh runtimes, fixed settings except 1000 versus 300 steps, validation-only stopping, and explicit limitations after prior test exposure. It then asks learners to save validation and selected-step tables outside the first runtime, without providing a dedicated pre-test export/download cell or named comparison-file schema. The canonical report is generated only much later.

**Improvement:** add a small validation-only activity export carrying model identity, settings, selected step, metrics and a digest. Provide one later comparison cell that checks matching controls and joins rows by model. Reusing verified frozen feature artifacts could reduce repeated setup in an explicitly designed advanced workflow; it should not be improvised in the current shared workspace.

The local 300-versus-1000-step synthetic probe had identical checkpoint histories through step 300. The larger budget selected lower validation loss on that fixture. This verifies only the loop-level change-one-thing mechanism, not both real-model hosted activity runs.

### ICR-08 — Historical selection explanations should remain conditional

**Severity:** Minor wording improvement.

Section 10 appropriately warns about selection at the first checkpoint or step cap. Its historical narrative says the learning-rate change lets each probe reach its validation minimum at its own step, while the release record reports ConvNeXt and ViT selected at the 1000-step cap in a historical run. Do not imply that an interior or global optimum was established for every model. The newer activity text correctly says a cap selection does not prove loss was still improving; keep that more careful explanation throughout.

Source for this section: [gallery/test cells](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tools/multimodel_image_classification_workshop_source.py#L180-L199); [BYOD, export and activity](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/tools/multimodel_image_classification_workshop_source.py#L200-L253); [historical probe-selection record](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/c92326be7b2754e7c00cf5cdb1c9a402cb3f16c7/docs/release-verification.md#multi-model-image-classification-workshop).

## 5. Promise-to-evidence trace

| Promise / objective | Implementation / observed evidence | Review conclusion |
|---|---|---|
| Compare pretrained systems on the same task | One validated dataset, fixed splits, native model transforms, frozen feature extraction and common probe policy. STANDARD execution reported by the repository. | Sound stated comparison; require explicit feature/prediction identity checks to protect it. Do not demand identical native transforms or claim pure architecture isolation. |
| Explain frozen backbone versus trained head | Feature runner uses evaluation/inference mode; the optimizer receives only new linear-head parameters. Introductory diagram, Section 10 and glossary explain the distinction. | Supported by source. No model-backed feature extraction independently repeated here. |
| Reuse training-only normalization | Means/stds computed with the training mask and exported with the head. | Source and synthetic-loop positive control agree. |
| Compare against a majority baseline | Training labels determine the fixed rule; Section 6 reports validation and Section 14 later reports test. | No early test-baseline defect found. Synthetic six-class arithmetic gives 1/6 accuracy and 1/21 macro-F1. |
| Select a checkpoint without using test metrics | Training/validation masks and validation-loss checkpoints; test scoring occurs later. | Supported for the intended execution order; selection is bounded to observed checkpoints. |
| Reconstruct serialized predictions | Head and normalization tensors are saved/reloaded; test runner starts separately. | Positive synthetic reload difference 0.0. Manifest/cache binding is incomplete. This does not validate the full image-to-prediction serving boundary. |
| Inspect error patterns | Confusion matrices, disagreement table and rule-based gallery. | Meaningful design; caption overlap obstructs the gallery, and missing-row joins need protection. |
| Make one controlled change | Two fresh runtimes; only probe-step budget changes; stop before freeze/test. | Coherent activity. Local loop-prefix check passed; user handoff and hosted activity execution remain to verify. |
| Use own compatible data | Directory/ZIP branch, deterministic per-class split and downstream common workflow. | Real implemented path, but numeric/missing-token/long-label failures need repair; full model-backed qualification remains open. |
| Retain a reproducible record | Dataset/feature caches, predictions, artifacts, metrics, figures and provenance exported. | Useful audit material; improve current-run inventory and exact runtime/notebook identity. |

## 6. What is not a finding

**Native preprocessing differences are not silently introduced.** The notebook expressly identifies them as part of each pretrained system. Enforcing one arbitrary transform would answer a different question.

**Extracting test features early is not by itself test leakage here.** The inspected extractor uses a frozen evaluation-mode backbone; the normalizer is fitted on training features; the probe's optimization and selection use train/validation masks. This review did not establish that test images update fitted parameters. The early representative gallery uses training images, unlike an EDA plot that reveals test outcomes.

**The held-out feature preview is not claimed to train a new classifier.** Section 17 displays already reloaded test predictions and says they are new relative to adaptation. A feature-free, unlabeled-image inference example would help practical reuse, but its absence is not automatically an INF2 failure. Complete downstream reconstruction/serving compatibility remains unverified.

**Small test size is appropriately disclosed.** A 48-image test has 1/48 ≈ 2.08 percentage points per image. Observer overlap and different pretraining recipes are acknowledged. These caveats should remain, and should not be used to pretend the evaluation-boundary defects are harmless.

**Existing positive tests and Colab receipts still count for their scopes.** Acquisition-only tests do not close a prediction-CSV label bug; fault injection does not prove the historical default output was malformed. The framework requires both distinctions.

## 7. Targeted specification alignment

The notebook declares specification 2.1. The currently inspected fleet document is 2.2. The following are targeted checks against the read sections, not a complete version-migration audit:

| Requirement area | Assessment |
|---|---|
| Profile/mode, standalone workflow and default sample | Declared and supported by inspected source; execution evidence attributed to its recorded source/path. |
| GDL1–GDL5, GDL7–GDL12, GDL14 | Substantial orientation, objectives, conceptual stages, predictions, worked checkpoints, infrastructure separation and conclusion prompts. The gallery and activity handoff need improvement. |
| DAT12 / VAL6 | Move explicit BYOD bounds to the pre-upload instructions. |
| DAT13–DAT14 / DAT19 / VAL7 | ICR-01 demonstrates accepted labels failing later or truncating without disclosure. |
| SPL6–SPL8 / FT2–FT5 | The inspected split/normalization/linear-probe sequence supports these intended mechanisms. No full conformance certificate implied. |
| EVAL2 / EVAL8 / OUT4 | ICR-02 requires complete, authoritative row/label/probability validation; ICR-03 requires semantic inference-state binding. |
| UNC2 | Softmax scores are explicitly not calibrated correctness probabilities. Retain this explanation. |
| ART2 / ART5 / OUT8 / VER2–VER4 | Real file reload exists, but the complete semantic boundary needs stronger binding and downstream reconstruction remains unverified. |
| ART6 | Whole-directory exports can carry prior-run state under noncanonical reuse; define intended bundle membership. |
| Release evidence | Preserve the documented STANDARD run. Do not infer FULL, DINOv2, all-label BYOD or repaired-revision qualification from it. |

Specification: [NOTEBOOK_SPEC.md](https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md), version 2.2 / blob recorded above. Mandatory conformance and finding severity are separate: a small pre-upload contract omission can still need correction even though it is not the most consequential technical defect.

## 8. Remediation and readiness gates

First repair label preservation and evaluator validation together, since they meet at the same prediction boundary. Then bind the manifest/cache/runner identities in the frozen experiment. Repair the gallery layout and pre-upload contract before guided use. Retain the existing source/version-labelled historical evidence.

The release check should exercise the repaired default STANDARD run in a fresh T4 runtime, representative BYOD with literal string labels and numeric-looking labels, a documented invalid-input rejection, and the two-runtime activity. Qualify FULL and DINOv2 separately. Include a fault-injection regression suite for missing/duplicate IDs, modified truth/probabilities, class-order changes and cached-feature changes. Tests of corrected behavior should execute the actual repaired notebook cells, not just the fixed historical excerpts in this review package.

A short learner observation should ask a representative participant to explain the image → frozen representation → trained probe sequence; interpret accuracy versus log-loss; identify a class-specific error from the gallery; make the bounded step-budget change; save the pre-test comparison; and explain the limits of a small, photographer-overlapping sample. This review did not measure learning outcomes.

**Final readiness:** Needs revision for the advertised combined default/interactive/BYOD experience. No full default-run failure was independently established here. No repository changes were made.

## 9. Evidence package

Companion archive: `DIMER_Image_Classification_Review_Probes.zip`.

The 13 grouped checks, source-excerpt harness, local package inventory and synthetic gallery rendering are included. Assertions in that package intentionally document the reviewed behavior, including defects. Passing those probes is not equivalent to passing release acceptance tests.

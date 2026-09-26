# DIMER Workshop Specification: Comparing Modern Image Classification Architectures

**Status:** Proposed  
**Notebook specification:** DIMER `NOTEBOOK_SPEC` **2.1**  
**Notebook profile:** `E2E`  
**Pedagogical mode:** `WORKSHOP`  
**Proposed filename:** `DIMER_MultiModel_Image_Classification_Workshop.ipynb`  
**Canonical runtime:** NVIDIA Tesla T4 or equivalent  
**Canonical execution:** standalone, credential-free, top-to-bottom `Run all`

---

# 1. Purpose

This workshop compares several live DIMER image-classification architectures on the **same six-class transfer-learning problem**.

Core models:

| Architecture | DIMER model | Upstream checkpoint |
|---|---|---|
| Residual CNN | **ResNet-50** | `timm/resnet50.a1_in1k` |
| Efficient CNN | **MobileNetV4-Conv-Small** | `timm/mobilenetv4_conv_small.e2400_r224_in1k` |
| Modern ConvNet | **ConvNeXt-Tiny** | `timm/convnext_tiny.in12k_ft_in1k` |
| Plain Vision Transformer | **ViT-B/16** | `timm/vit_base_patch16_224.orig_in21k_ft_in1k` |
| Hierarchical Transformer | **SwinV2-Tiny** | `timm/swinv2_tiny_window8_256.ms_in1k` |
| High-resolution Transformer | **EVA-02 Base 448** | `timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` |

The central question is:

> Given the same labelled images and the same downstream classification task, how useful are the pretrained representations of different modern vision architectures, and what accuracy/compute tradeoffs do they make?

The workshop MUST NOT present results as a general ranking of architectures.

---

# 2. Why `E2E`

The workshop performs real supervised adaptation:

```text
pretrained backbone
→ frozen feature extraction
→ train a new six-class linear head
→ validation selection
→ held-out test
→ export adapter
→ fresh reload
→ new-image inference
```

The adaptation artifact is the new classification head plus the preprocessing statistics necessary to reproduce inference.

Therefore:

**Profile:** `E2E`

rather than `TASK-INFERENCE`.

---

# 3. Why use a common linear probe

The individual DIMER repositories expose different fine-tuning policies:

- some perform full or broad gradient fine-tuning;
- ViT uses a frozen probe followed by bounded transformer-block unfreezing;
- Swin uses its own artifact/fine-tuner contract;
- input resolutions and pretrained transforms differ.

Using those model-specific policies directly would confound:

**architecture quality**

with:

**different optimization recipes**.

The canonical workshop therefore uses one controlled adaptation policy across every architecture:

> **freeze the pretrained backbone, extract its native penultimate representation, then train the same six-class linear probe.**

This isolates the practical quality of each pretrained representation while keeping training inexpensive and reproducible.

The individual DIMER E2E notebooks remain authoritative for full model-specific fine-tuning.

---

# 4. Core model identities

## ResNet-50

**Model:** `timm/resnet50.a1_in1k`  
**Revision:** `767268603ca0cb0bfe326fa87277f19c419566ef`  
**Architecture:** residual CNN  
**Approx. parameters:** 25.6M  
**Native input:** 224×224  
**License:** Apache-2.0

---

## MobileNetV4-Conv-Small

**Model:** `timm/mobilenetv4_conv_small.e2400_r224_in1k`  
**Revision:** `331fb803779522b685cf942e15f914fb6741c1eb`  
**Architecture:** compute-efficient convolutional network  
**Approx. parameters:** 3.8M  
**Native input:** 224×224  
**License:** Apache-2.0

---

## ConvNeXt-Tiny

**Model:** `timm/convnext_tiny.in12k_ft_in1k`  
**Revision:** `aa096f03029c7f0ec052013f64c819b34f8ad790`  
**Architecture:** modernized ConvNet  
**Approx. parameters:** 28.6M  
**Native input:** 224×224  
**License:** Apache-2.0

---

## ViT-B/16

**Model:** `timm/vit_base_patch16_224.orig_in21k_ft_in1k`  
**Revision:** `e0bd370de6799e8d1f47a911174ff4c3708e2323`  
**Architecture:** global self-attention over 16×16 image patches  
**Approx. parameters:** 86.6M  
**Native input:** 224×224  
**License:** Apache-2.0

---

## SwinV2-Tiny

**Model:** `timm/swinv2_tiny_window8_256.ms_in1k`  
**Revision:** `650d02aabf05e8adbd060a739ab39e39f53da639`  
**Architecture:** hierarchical/windowed transformer  
**Native input:** 256×256  
**License:** MIT

---

## EVA-02 Base 448

**Model:** `timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`  
**Revision:** `81063ecfe9c381a16a19d06f396d6c7011aa426a`  
**Architecture:** high-resolution transformer  
**Approx. parameters:** 87.1M  
**Native input:** 448×448  
**License:** MIT

---

# 5. Execution tiers

To keep the default workshop practical:

```python
WORKSHOP_TIER = "STANDARD"  # @param ["STANDARD", "FULL"]
```

## STANDARD

Runs:

- MobileNetV4
- ResNet-50
- ConvNeXt-Tiny
- ViT-B/16

This gives a clean progression:

```text
efficient CNN
→ classic residual CNN
→ modern ConvNet
→ plain transformer
```

This SHOULD be the canonical `Run all` release path.

## FULL

Adds:

- SwinV2-Tiny
- EVA-02 Base 448

This expands the transformer comparison to:

```text
plain transformer
→ hierarchical/windowed transformer
→ large-scale high-resolution transformer
```

Both modes require separate clean-runtime qualification.

---

# 6. Canonical dataset

The workshop SHALL reuse the existing labelled corpus already used by:

- `vit-classification-pipeline`; and
- `dinov2-feature-extraction-pipeline`.

Dataset:

**180 CC0 iNaturalist bird photographs**

Structure:

- **6 bird species**
- **30 photographs per species**
- every image pinned by photo ID, byte size, and SHA-256
- fetched at runtime from iNaturalist Open Data
- no photographs redistributed by the repository
- images are wildlife photographs rather than photographs of people

The workshop SHOULD carry the existing sample manifest directly into the notebook.

---

# 7. Existing split

Reuse the currently qualified split:

| Split | Total | Per class |
|---|---:|---:|
| Train | **108** | 18 |
| Validation | **24** | 4 |
| Test | **48** | 8 |
| Total | **180** | 30 |

Split seed:

```text
42
```

The same photograph MUST appear in only one split.

The existing corpus includes observer overlap across splits. This MUST be disclosed as a limitation because photographer/camera style may contribute to learned signals.

The workshop MUST characterize all results as **sample-sanity evidence**, not species-recognition benchmark evidence.

---

# 8. Why this dataset is particularly useful

The corpus intentionally includes species that are not all represented directly by ImageNet-1k labels.

That means the workshop is not simply asking:

> Which ImageNet classifier remembers its original labels?

Instead it asks:

> How transferable is each pretrained visual representation when given a new small label space?

This is precisely where comparing modern pretrained architectures becomes educational.

---

# 9. Dataset contract

Canonical record:

```text
{
  "id": string,
  "image": PIL RGB image,
  "label": string
}
```

Requirements:

- unique ID;
- valid decodable image;
- RGB conversion supported;
- exactly one class label per image;
- six canonical classes in the built-in fixture;
- no duplicate image bytes across splits;
- image side ≤4,096 px;
- no silent image omission.

---

# 10. Preprocessing fairness

All architectures MUST receive the **same original JPEG files**.

However, each architecture MUST use its **own pinned pretrained evaluation transform**.

Examples:

- ResNet-50: model-native resize + 224 crop;
- MobileNetV4: model-native 224 transform;
- ConvNeXt: model-native 224 transform;
- ViT: model-native 224 transform;
- SwinV2: 256×256 contract;
- EVA-02: 448×448 squash-resize contract.

The workshop MUST NOT force every architecture into one arbitrary 224×224 transform merely to make dimensions match.

Model-specific preprocessing is part of the pretrained model contract.

The comparison therefore controls:

> original images and downstream task

while preserving:

> architecture-specific preprocessing required by the pretrained checkpoint.

---

# 11. Architecture primer

Before running models, briefly explain each family.

### ResNet

Core idea:

```text
convolution + residual connections
```

Question:

> Can a deep CNN learn better representations by learning residual corrections instead of complete transformations?

---

### MobileNetV4

Core idea:

```text
accuracy under compute constraints
```

Question:

> How much classification quality can be retained with a much smaller network?

---

### ConvNeXt

Core idea:

```text
modernize CNN design using lessons from transformers
```

Question:

> How competitive can a pure convolutional model remain after modern architectural updates?

---

### ViT

Core idea:

```text
split image into patches
→ treat patches as tokens
→ global self-attention
```

Question:

> Can an image be modelled effectively as a sequence?

---

### SwinV2

Core idea:

```text
local/windowed attention
→ shifted windows
→ hierarchical feature maps
```

Question:

> Can transformer attention be made more spatially structured and efficient?

---

### EVA-02

Core idea:

```text
large-scale transformer pretraining
+ high-resolution visual input
```

Question:

> What does a heavier, higher-resolution transformer buy us?

---

# 12. Phase A — inspect the dataset

The notebook SHOULD show:

- class counts;
- image dimensions;
- representative images from each species;
- train/validation/test counts;
- observer overlap;
- duplicate-content check.

Exercise:

> Before running the models, which species pairs look visually hardest to separate?

This prediction can later be compared with the confusion matrices.

---

# 13. Phase B — majority baseline

Because the built-in corpus is class-balanced:

\[
Accuracy_{majority} = \frac{1}{6}
\]

approximately:

```text
16.7%
```

The notebook MUST calculate this from the actual training split rather than hard-code it as the evaluation result.

Report:

- accuracy;
- macro-F1.

This is the minimum meaningful reference point.

---

# 14. Phase C — pretrained feature extraction

For each architecture:

1. resolve pinned model snapshot;
2. verify immutable revision and weight digests;
3. reconstruct using upstream `timm`;
4. apply model-native preprocessing;
5. remove or bypass the ImageNet classification head;
6. extract the model's native pooled penultimate feature vector;
7. export features for all train/validation/test images;
8. unload the backbone before loading the next architecture.

Output per architecture:

```text
id
split
label
feature_vector
feature_dimension
model
```

Vectors MUST remain associated with image IDs.

---

# 15. Model-memory strategy

Only one pretrained backbone SHOULD be resident on GPU at any time.

Canonical cycle:

```text
load backbone
→ extract train features
→ extract validation features
→ extract test features
→ write feature cache
→ unload backbone
→ clear GPU cache
→ load next architecture
```

This prevents unnecessary GPU memory accumulation.

---

# 16. Common feature preprocessing

The linear probe MAY standardize feature dimensions, but fitted preprocessing MUST use training features only.

For architecture \(m\):

\[
z = \frac{x-\mu_{train,m}}{\sigma_{train,m}}
\]

Training mean/std are then reused for:

- validation;
- test;
- new-image inference.

The notebook MUST NOT fit normalization on validation/test features.

Near-zero standard deviations MUST be handled explicitly and recorded.

---

# 17. Common adaptation policy

Every architecture receives exactly the same linear-probe training policy.

Recommended default:

```text
classifier: Linear(feature_dim, 6)
optimizer: AdamW
learning rate: 1e-2
weight decay: 1e-4
maximum steps: 300
seed: 0
loss: cross-entropy
backbone: frozen
```

Probe training occurs on:

**108 training images**

Validation occurs on:

**24 validation images**

No test image is used for training or selection.

---

# 18. Probe selection

The probe SHOULD be checkpointed at fixed intervals.

Recommended:

```text
evaluate validation log-loss every 10 steps
```

Selection criterion:

> lowest validation log-loss

The selected probe is frozen before the test split is opened.

The notebook MUST record:

- selected step;
- training loss;
- validation log-loss;
- accuracy;
- macro-F1.

---

# 19. Freeze-before-test

After all probes have been selected, write:

```text
outputs/frozen/frozen_experiment.json
```

It MUST contain:

- dataset digest;
- class ordering;
- split identities;
- model IDs;
- immutable revisions;
- pretrained transforms;
- feature dimensions;
- training-feature normalization statistics;
- probe hyperparameters;
- selected training step;
- selected probe weight digest;
- evaluation metric definitions.

The test section MUST load this record.

---

# 20. Final held-out evaluation

The final 48 photographs are scored exactly once under the frozen protocol.

Report per model:

- accuracy;
- macro-F1;
- log-loss;
- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrix.

All six architectures use:

- same images;
- same split;
- same labels;
- same classifier family;
- same probe optimizer;
- same selection metric.

---

# 21. Principal comparison table

Suggested table:

| Model | Params | Input | Feature dim | Accuracy | Macro-F1 | Log-loss | Feature extraction time |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV4 | … | 224 | … | … | … | … | … |
| ResNet-50 | … | 224 | … | … | … | … | … |
| ConvNeXt-Tiny | … | 224 | … | … | … | … | … |
| ViT-B/16 | … | 224 | … | … | … | … | … |
| SwinV2-Tiny | … | 256 | … | … | … | … | … |
| EVA-02 Base | … | 448 | … | … | … | … | … |

`STANDARD` omits Swin/EVA.

Parameter count and feature dimension SHOULD be read from the loaded model rather than typed manually into result records.

---

# 22. Architecture-efficiency comparison

Also compute:

### Accuracy per million parameters

For teaching only:

\[
\frac{accuracy}{parameters/10^6}
\]

This MUST NOT be treated as a definitive model-efficiency metric.

### Runtime

Measure:

- model load time;
- feature-extraction time;
- images/second;
- probe-training time;
- final classification time.

### GPU memory

Peak allocated GPU memory SHOULD be measured where practical.

This section allows students to examine:

> accuracy versus computational footprint

rather than accuracy alone.

---

# 23. Confusion analysis

For every architecture, display a confusion matrix.

Then build a cross-model error table:

```text
image_id
truth
mobilenet_prediction
resnet_prediction
convnext_prediction
vit_prediction
swin_prediction
eva_prediction
```

Useful analysis:

### Consensus successes

Images every model gets right.

### Consensus failures

Images every model misclassifies.

### CNN/transformer disagreements

Images where:

- CNNs agree;
- transformers agree;
- the two families differ.

### Architecture-specific successes

Images only one model classifies correctly.

This is more educational than simply comparing aggregate accuracy.

---

# 24. Error gallery

Show several held-out photographs selected using predefined rules:

1. unanimous correct;
2. unanimous incorrect;
3. largest model disagreement;
4. highest-confidence incorrect prediction if available;
5. one example from each class where practical.

Selection MUST be based on deterministic rules, not hand-picked after viewing results.

---

# 25. Artifact export

Each architecture produces a small transfer-classification artifact.

Suggested structure:

```text
artifacts/
└── <model-key>/
    ├── adapter.safetensors
    └── manifest.json
```

Artifact contents:

- linear probe weight;
- linear probe bias;
- feature normalization mean;
- feature normalization std.

Manifest:

```text
format
format_version
base_model_id
base_model_revision
base_weight_digest
feature_dimension
classes
class_order
training_configuration
selected_step
selection_metric
files
```

The artifact MUST NOT include the full pretrained backbone.

---

# 26. Fresh reload verification

For each architecture:

```text
unload probe
→ reconstruct adapter from serialized files
→ load base model
→ extract features from fixed verification images
→ apply stored train normalization
→ classify
```

Reloaded probabilities SHOULD match pre-export probabilities within explicit tolerance.

The notebook MUST verify more than file loading.

It MUST verify **output equivalence**.

---

# 27. New-image inference

After test evaluation and fresh reload, run each selected architecture on one or more new images that were not part of probe training.

Output:

```text
id
model
predicted_label
class_scores
```

Scores MUST be described as:

> softmax scores from the adapted six-class head

not calibrated probabilities.

---

# 28. Optional DINOv2 extension

DINOv2 SHOULD be included only as an optional **representation-learning baseline**, not as a core pretrained classifier.

**Model:** `timm/vit_small_patch14_dinov2.lvd142m`  
**Revision:** `4610ca143709d58a633b6397a74412c2c3842454`

Why separate it:

- DINOv2 has no pretrained 1,000-class head;
- its primary DIMER task is image feature extraction;
- it was self-supervised rather than ImageNet-label trained.

It MAY use the same:

- bird corpus;
- split;
- feature standardization;
- linear probe;
- evaluation protocol.

Display it separately as:

> **Self-supervised representation baseline**

This creates an especially useful teaching question:

> Does a dedicated self-supervised representation transfer better than features from supervised ImageNet classifiers?

---

# 29. Model-native fine-tuning extension

A later advanced section MAY demonstrate bounded model-native fine-tuning.

However it MUST NOT alter the canonical comparison.

Reason:

- individual model pipelines use different training contracts;
- a full-network update changes the quantity being compared;
- learning-rate and architecture differences make identical hyperparameters potentially unfair;
- runtime rises substantially.

If included, results SHOULD be labelled:

```text
model-native adaptation
```

and presented separately from:

```text
common frozen-probe comparison
```

---

# 30. Why not compare the original ImageNet heads?

The bird dataset contains a new six-species label space.

Some species are represented in ImageNet-1k, while others are not.

Direct ImageNet top-1 accuracy therefore cannot form a fair six-class evaluation.

The pretrained ImageNet head MAY be demonstrated for one or two examples as a conceptual exercise:

> What labels does the original classifier know?

But its predictions MUST NOT enter the six-class workshop leaderboard.

---

# 31. BYOD contract

BYOD SHALL use the existing DIMER image-classification dataset format.

Preferred archive:

```text
my_dataset.zip
├── labels.csv
├── image_001.jpg
├── image_002.jpg
└── ...
```

`labels.csv`:

```text
id,file,label
img001,image_001.jpg,class_a
img002,image_002.jpg,class_b
```

Required columns:

- `id`
- `file`
- `label`

---

# 32. BYOD validation

The notebook MUST validate:

- `labels.csv` exists;
- required columns exist;
- IDs are non-empty and unique;
- filenames resolve inside the archive/directory;
- duplicate file aliases are handled explicitly;
- every image decodes successfully;
- class labels are non-empty;
- 2–100 classes;
- sufficient examples exist for splitting;
- image sides are within supported ceilings;
- duplicate image content across splits is detected;
- train/validation/test class coverage is reported.

ZIP extraction, if used, MUST reject:

- absolute paths;
- `..` traversal;
- symlinks.

Prefer decoding directly from archive bytes where practical.

---

# 33. BYOD split

If user data do not already define splits:

use a deterministic stratified split.

Recommended:

```text
60% train
15% validation
25% test
seed = 42
```

Where possible, class coverage MUST be preserved.

The notebook MUST explain that random image splitting assumes images are sufficiently independent.

For datasets with:

- subjects;
- photographers;
- locations;
- video sequences;
- camera traps;
- patients;

the user SHOULD instead provide a grouping boundary.

---

# 34. BYOD adaptation path

Because this is `E2E`, BYOD MUST run:

```text
validate
→ split
→ model-specific preprocessing
→ feature extraction
→ fit training normalization
→ linear probe
→ validation selection
→ test evaluation
→ adapter export
→ fresh reload
→ inference
```

BYOD MUST NOT degrade into inference-only operation.

---

# 35. BYOD privacy guidance

The notebook MUST state:

> User-supplied images are processed inside the selected notebook runtime and are not submitted to DIMER workers or APIs. A hosted notebook remains an external compute environment. Do not upload confidential, personal, restricted, regulated, biometric, or proprietary imagery unless you are authorized to process it there.

---

# 36. Common evaluator

The workshop SHALL own one common evaluator.

For classes:

```text
C = [c1, c2, ..., c6]
```

compute:

### Accuracy

\[
accuracy = \frac{\text{correct predictions}}{\text{all predictions}}
\]

### Precision per class

### Recall per class

### F1 per class

### Macro-F1

\[
MacroF1 =
\frac{1}{|C|}
\sum_{c \in C}F1_c
\]

### Multiclass log-loss

using the six-class probe softmax.

Class ordering MUST remain explicit in exported probability arrays.

---

# 37. Runtime isolation

The models currently share similar `timm`-based ecosystems but not necessarily identical dependency pins.

The workshop SHOULD either:

### Preferred

derive one tested common compatible environment;

or, if dependency convergence is unsafe:

### Fallback

use isolated subprocess environments as in the time-series workshop.

A model MUST NOT silently run under an untested incompatible package version simply to keep the notebook shorter.

---

# 38. Standard notebook anatomy

```text
0. Workshop introduction
1. Runtime and execution tier
2. Architecture primer
3. Model identities and supply-chain pins
4. Acquire the iNaturalist corpus
5. Validate images and split
6. Explore the dataset
7. Majority baseline
8. Common feature-extraction contract
9. MobileNetV4
10. ResNet-50
11. ConvNeXt-Tiny
12. ViT-B/16
13. SwinV2 [FULL]
14. EVA-02 [FULL]
15. Common linear probes
16. Validation comparison
17. Freeze experiment
18. Held-out test
19. Architecture comparison
20. Confusion/error analysis
21. Computational tradeoffs
22. Export adapters
23. Fresh reload verification
24. New-image inference
25. BYOD
26. Optional DINOv2
27. Optional model-native fine-tuning
28. Interpretation and limits
29. Troubleshooting
30. Glossary
```

---

# 39. Workshop exercises

## Exercise A — architecture prediction

Before running models:

> Which architecture do you expect to produce the strongest transferable features?

Ask learners to justify their prediction based on:

- model size;
- pretraining;
- CNN versus transformer;
- input resolution.

---

## Exercise B — efficiency

After evaluation:

> Did the model with the most parameters perform best?

Then compare:

- accuracy;
- macro-F1;
- weight size;
- feature extraction time.

---

## Exercise C — resolution

Compare EVA-02's 448×448 input against 224/256-pixel models.

Ask:

> Did the additional visual resolution materially improve this task?

---

## Exercise D — error analysis

Show an image where architecture predictions disagree.

Ask:

> What visual features might explain the disagreement?

---

## Exercise E — transfer

Ask:

> If this were a Philippine biodiversity classification project, what data would you need before treating these model differences as meaningful?

Expected considerations include:

- local species;
- geography;
- photographers/cameras;
- habitats;
- seasons;
- class imbalance;
- train/test grouping.

---

# 40. Computational tradeoff table

The workshop SHOULD produce:

| Model | Architecture | Params | Native input | Weight size | Feature extraction time | Images/s | Peak VRAM | Test accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

The table MUST NOT generate a single composite score or overall winner.

Students should interpret tradeoffs based on deployment requirements.

---

# 41. Interpretation boundaries

The notebook MUST state:

### Small sample

48 test photographs are too few for general performance claims.

One image represents roughly 2.1 percentage points of total accuracy.

### Observer overlap

The existing fixture prevents duplicate-image leakage but has photographer overlap across splits.

Photographic style may therefore contribute some transferable signal.

### Pretraining differences

The architectures were pretrained using different:

- datasets;
- objectives;
- resolutions;
- recipes.

This is a comparison of **available pretrained systems**, not an architecture-only controlled scientific experiment.

### Input resolution

Performance differences may partly arise from:

- crop policy;
- resize method;
- resolution;
- aspect-ratio handling.

Those are part of each deployed checkpoint.

### Softmax is not confidence calibration

Probe softmax values MUST NOT be presented as guaranteed correctness probabilities.

### Dataset domain

Wildlife photographs do not establish performance on:

- products;
- medical images;
- satellite imagery;
- documents;
- industrial cameras;
- faces;
- local biodiversity deployment data.

---

# 42. Output structure

```text
outputs/
├── data/
│   ├── dataset_manifest.json
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── features/
│   ├── mobilenet/
│   ├── resnet50/
│   ├── convnext/
│   ├── vit/
│   ├── swin/
│   └── eva02/
├── validation/
│   ├── metrics.csv
│   └── predictions.csv
├── frozen/
│   └── frozen_experiment.json
├── test/
│   ├── aggregate_metrics.csv
│   ├── per_class_metrics.csv
│   ├── predictions.csv
│   └── disagreements.csv
├── artifacts/
│   └── <model-key>/
├── figures/
├── provenance/
│   ├── models.json
│   └── experiment_manifest.json
└── workshop_summary.json
```

Folders for models not selected in `STANDARD` MAY be absent.

---

# 43. Provenance

`experiment_manifest.json` SHOULD include:

```text
notebook_spec
notebook_profile
notebook_mode
workshop_revision
execution_tier
timestamp

dataset:
  source
  sample_manifest_revision
  license
  dataset_digest
  class_names
  split_seed
  split_counts
  observer_overlap

models:
  model_id
  revision
  license
  pretrained_transform
  parameters
  native_input_size
  feature_dimension
  checkpoint_digest

adaptation:
  method = linear probe
  normalization = train-feature z-score
  optimizer
  learning_rate
  weight_decay
  maximum_steps
  selection_metric
  seed

evaluation:
  metrics
  majority_baseline
  test_sample_count
```

---

# 44. Notebook metadata

```json
{
  "dimer": {
    "notebook_spec": "2.1",
    "notebook_profile": "E2E",
    "notebook_mode": "WORKSHOP",
    "standalone": true,
    "capability": "multi-model-image-classification",
    "carrier": "common-probe comparative transfer-classification workshop",
    "dataset": "six-species CC0 iNaturalist bird corpus",
    "default_tier": "STANDARD",
    "canonical_runtime": "NVIDIA Tesla T4",
    "worker_required": false,
    "credentials_required": false,
    "clean_runtime_evidence": "pending"
  }
}
```

---

# 45. Release acceptance

| Requirement | STANDARD | FULL |
|---|---:|---:|
| Notebook Spec 2.1 | PASS | PASS |
| Fresh `Run all` | PASS | PASS |
| Automatic sample acquisition | PASS | PASS |
| Dataset digest verification | PASS | PASS |
| No Git clone | PASS | PASS |
| No DIMER source fetch | PASS | PASS |
| No DIMER worker | PASS | PASS |
| No credentials | PASS | PASS |
| Majority baseline | PASS | PASS |
| MobileNetV4 | PASS | PASS |
| ResNet-50 | PASS | PASS |
| ConvNeXt-Tiny | PASS | PASS |
| ViT-B/16 | PASS | PASS |
| SwinV2 | N/A | PASS |
| EVA-02 | N/A | PASS |
| Common probe adaptation | PASS | PASS |
| Validation selection | PASS | PASS |
| Freeze-before-test | PASS | PASS |
| Independent test | PASS | PASS |
| Per-class evaluation | PASS | PASS |
| Artifact export | PASS | PASS |
| Fresh reload parity | PASS | PASS |
| BYOD positive case | PASS | PASS |
| BYOD refusal case | PASS | PASS |
| Machine-readable outputs | PASS | PASS |

---

# 46. Suggested registry entry

```markdown
| Notebook | Profile | Mode | Capability | Runtime | Sample | BYOD | Run-all | Status |
|---|---|---|---|---|---|---|---|---|
| `DIMER_MultiModel_Image_Classification_Workshop.ipynb` | `E2E` | `WORKSHOP` | common-probe transfer classification across modern CNN/transformer architectures | T4 | 180 CC0 iNaturalist bird photographs / six species | yes | pending | candidate |
```

---

# 47. Implementation principle

The notebook must compare architectures under one controlled downstream task:

> **same images → model-native preprocessing → pretrained representation → same linear probe → same split → same metrics → same test set**

That keeps the experiment interpretable.

The workshop SHOULD avoid becoming:

> six unrelated model tutorials pasted into one notebook.

The individual DIMER repositories already provide that model-specific depth.

The comparative notebook adds the missing fleet-level question:

> **What practical differences emerge when CNNs, efficient ConvNets, plain vision transformers, hierarchical transformers, and high-resolution transformers are asked to transfer to the same small image-classification problem?**
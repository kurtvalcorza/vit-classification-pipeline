---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-classification
base_model: timm/vit_base_patch16_224.orig_in21k_ft_in1k
date_published: "2022-12-22"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/timm/vit_base_patch16_224.orig_in21k_ft_in1k)"
---

# ViT-B/16 orig_in21k_ft_in1k — Image Classification

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fvit__base__patch16__224.orig__in21k__ft__in1k-ffcc4d?style=flat)](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k_ft_in1k)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-huggingface%2Fpytorch--image--models-181717?style=flat&logo=github&logoColor=white)](https://github.com/huggingface/pytorch-image-models)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2010.11929-b31b1b.svg)](https://arxiv.org/abs/2010.11929)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, resolve and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/vit-classification-pipeline/blob/main/tutorials/vit_classification_colab.ipynb) [`vit_classification_colab.ipynb`](https://github.com/kurtvalcorza/vit-classification-pipeline/blob/main/tutorials/vit_classification_colab.ipynb)  
  *ImageNet-1k single-label classification with the pinned `timm/vit_base_patch16_224.orig_in21k_ft_in1k` weights: one normalized 3×224×224 tensor → 196 patch tokens + CLS → 1000 logits → softmax, argmax and top-k; no adaptation occurs.*

---

#### Description

`timm/vit_base_patch16_224.orig_in21k_ft_in1k` is the original Vision Transformer ViT-B/16 checkpoint from "An Image is Worth 16x16 Words" (Dosovitskiy et al., arXiv:2010.11929): per the pinned upstream README it was "Trained on ImageNet-21k and fine-tuned on ImageNet-1k in JAX by paper authors, ported to PyTorch by Ross Wightman", published in the `timm` library and pinned here to revision `e0bd370de6799e8d1f47a911174ff4c3708e2323`. Architecturally it is a plain transformer encoder over image patches: a 16×16 strided convolution (`patch_embed.proj`) turns a 224×224 image into 196 patch embeddings of width 768, a learned class token is prepended, learned position embeddings are added, 12 pre-norm multi-head self-attention blocks follow, and a 1000-way linear head reads the class token (`global_pool: token`, CLS pooling) — 86.6 M parameters and 16.9 GMACs at 224 px per the upstream README. At inference the network maps one normalized 3×224×224 tensor to 1000 logits in a single forward pass; no adaptation, fine-tuning, or in-context conditioning happens in this repository. What this repository adds is packaging: the `ViTClassificationPipeline` class in `src/vit_classification_pipeline/pipeline.py`, digest verification of the local snapshot (`verify_snapshot`), staged download by immutable revision (`stage_missing_files`), input validation with named ceilings, a fixed output contract, the `top_k_accuracy` metric helper, and the `validate_inputs` / `evaluation_report` stage helpers the standalone tutorial calls.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is single-label image classification: input one PIL image or a batch of up to `MAX_BATCH = 64` images; output, per image, the `top_k` (default 5) ImageNet-1k classes with their softmax scores plus the argmax label. Envisioned applications are general object and scene tagging on photographs that resemble ImageNet — consumer photos, product images, wildlife camera frames, stock-image indexing — and use as the transformer-family reference point next to the sibling CNN pipelines (ResNet-50, ConvNeXt-Tiny, MobileNetV4) and the higher-resolution EVA-02 pipeline, so that a reader can compare architectures at the same 1000-class task. In a larger system the pipeline is meant as an inference component or a baseline, not as a decision engine; the 768-d class-token features are not exposed by this package (the DINOv2 sibling covers feature extraction).

###### Primary Intended Users

The intended users are machine-learning engineers, data scientists and application developers integrating a classifier into research prototypes, internal enterprise tooling, or the DIMER model workbench. The pipeline assumes its users understand that the label space is fixed to the 1000 ImageNet-1k classes, that a softmax score is not a calibrated probability, that the input is resized and center-cropped to a fixed 224×224 so large or oddly framed images lose detail and edge content, that images far from the ImageNet distribution produce confident-looking nonsense, and that any deployment on their own data needs a labelled evaluation set. It is not designed for hobbyist "point and trust" use.

###### Out-of-scope use cases

1. **Capability boundary:** not object detection, segmentation, multi-label tagging, OCR, or open-vocabulary classification; anything outside the 1000 ImageNet-1k classes cannot be named. Feature extraction is not exposed here — use `dinov2-feature-extraction-pipeline`.
2. **Input boundary:** only PIL images are accepted (`TypeError` otherwise); any side above `MAX_IMAGE_SIDE = 4096` px or below 1 px is rejected; batches above 64 are rejected. The checkpoint declares `fixed_input_size: true` at 3×224×224 (`INPUT_SIZE = 224`): every image is resized so its shorter side is 248 px and center-cropped to 224×224 (`crop_pct = 0.9`, bicubic) by the timm eval transform, and there is no dynamic-resolution or position-embedding-interpolation path in this package, so a 4096-px image is reduced 18× before the model sees it. Non-RGB modes are converted to RGB; depth, multispectral and video inputs are unsupported.
3. **Decision boundary:** not for autonomous or high-impact decisions — content moderation takedowns, safety interlocks, medical or forensic triage — without a human reviewing the prediction and a locally measured error rate.

#### Factors

###### Groups

The pipeline is not human-centric: it is an object-centric classifier whose label space contains no person-identity, age, gender or skin-type categories. ImageNet-1k and the ImageNet-21k pretraining set nevertheless contain many images of people, and the ImageNet hierarchy has documented label problems (ambiguous, offensive and mislabelled categories). Neither the upstream `timm` card nor this repository reports any group-level performance breakdown, and neither training set is group-audited. The fairness audit therefore transfers to the operator: before deployment, measure `top_k_accuracy` on a labelled sample of your own data stratified by the groups that matter to your application, and treat any material gap as a blocker.

###### Instrumentation

ImageNet-1k and ImageNet-21k images were collected from web image searches (Deng et al., 2009) and are consumer camera photographs of varied, undocumented provenance — many makes of camera, lens and post-processing, mostly JPEG-encoded. The pipeline consumes decoded pixel arrays, so the instrument sits behind PIL: resolution, JPEG compression level, colour profile, white balance and sensor noise all reach the model as changed pixel statistics after the 248-px resize and 224-px crop, and the 16×16 patch grid means a one-pixel shift moves content between patches. The pipeline does not detect drift, blur, over-exposure or a change of capture device; it only rejects non-image types and images outside the 1–4096 px side range. Operators with a fixed camera should validate on frames from that camera.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `timm==1.0.29`, `pillow==11.3.0` (exact pins in `pyproject.toml`). CUDA is optional; `from_pretrained` picks `cuda:0` when available, else CPU, and runs in float32 on both. On this repository's smoke run (Windows venv, CPU only with `CUDA_VISIBLE_DEVICES=-1` and `device="cpu"`, 346 MB snapshot) loading the verified snapshot took 3.19 s, the first 224-px prediction 0.22 s including transform and warm-up, and a second prediction 0.06 s; the CUDA path has not been executed by this repository. Data environment: inputs are assumed to be natural photographs whose subject is one of the 1000 classes, framed roughly as in ImageNet and tolerant of the 10 % border removed by the center crop; line drawings, medical scans, satellite tiles, heavy occlusion or unusual viewpoints fall outside that assumption and degrade accuracy in ways the pipeline does not measure.

#### Metrics

###### Performance Measures

The only measure the code reports is `top_k_accuracy(predictions, targets, k)` in `pipeline.py`, surfaced through `evaluation_report` at k=1 and k=5 when a caller supplies ground-truth ImageNet-1k indices (verdict `sample-sanity`) and withheld with verdict `not-measurable` otherwise. It captures discrete correctness of the ranking, which suits a 1000-way single-label classifier where the operational question is "is the right class first, or at least in the shortlist". It says nothing about calibration or per-class behaviour, so a reader using top-1 alone cannot tell whether errors are near-misses (fixable by a shortlist) or confident mistakes. The pinned upstream README reports no accuracy figure for this checkpoint (it points to the timm results tables instead), and this pipeline has not measured one; no ImageNet-1k accuracy is claimed here.

###### Decision thresholds

The default decision rule is `argmax` over the 1000 softmax scores, exposed as `DECISION_RULE = "argmax"` and reported as `predicted_index` / `predicted_label`; this is an implicit threshold of "highest score wins" with no minimum score. No acceptance threshold was set during development and none is shipped: the softmax score is uncalibrated, so any fixed cut-off would be arbitrary. A deployment that needs an abstain option must choose a score cut-off on its own labelled data, trading the cost of a wrong confident label (false positive) against the cost of an unanswered image (false negative) for its application.

###### Approaches to uncertainty and variability

This pipeline reports no accuracy number of its own, so there is no estimation procedure or dispersion to state, and the pinned upstream README carries no accuracy figure to qualify. When `evaluation_report` is given targets it labels each `top_k_accuracy` value "single sample, no dispersion estimate". Inference is deterministic given the same weights, device and library versions: there is no sampling, dropout is disabled by `model.eval()`, and no seed is required; small numeric differences between CPU, GPU and cuDNN or attention-kernel choices can reorder near-tied classes. The `score` field is a softmax over logits and is not calibrated; a caller who needs probabilities must fit a calibration map (for example temperature scaling) on their own labelled data.

#### Ethical considerations and biases

###### Data

The pinned upstream README discloses the training data as "Trained on ImageNet-21k and fine-tuned on ImageNet-1k in JAX by paper authors" (front matter `datasets: imagenet-1k, imagenet-21k`); the disclosure stops there — no per-image licensing, consent status or demographic composition is given, and both ImageNet sets are known to contain photographs of identifiable people scraped from the web, so the presence of personal data is not ruled out. This repository distributes code, tests, documentation and the generated tutorial notebook; the 346 MB `model.safetensors` snapshot is git-ignored and staged locally under `weights/vit-base-p16-224/` with a manifest, and no sample data is shipped. The operator must audit the images they submit for personal, confidential or proprietary content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone. Its only validation is the offline unit suite and the CPU smoke run in this repository. Where a sensitive use is foreseeable — for example flagging images in a moderation queue — it is admissible only with a human reviewer on every consequential outcome, an independent domain evaluation on representative data, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented and inspectable in `src/vit_classification_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest naming any other identity and fetches only absent entries at that revision; `verify_snapshot` re-hashes every file in `weights/vit-base-p16-224/dimer-base-manifest.json` and raises on the first size or SHA-256 mismatch before any weight is loaded; the Hub path is taken only with `allow_download=True` and then through timm's `hf-hub:<id>@<revision>` form; `trust_remote_code` is never enabled (timm executes no remote code). (2) Input integrity — `_check_inputs`, shared by `predict` and the public `validate_inputs`, rejects non-PIL inputs, empty or over-size batches, and images outside 1–4096 px before the model runs, so the tutorial's input manifest and the core method cannot diverge. (3) Reproducibility — exact `==` dependency pins, `model.eval()`, deterministic preprocessing from the snapshot's `pretrained_cfg`, and `model_id`/`model_revision` in every result, input manifest and evaluation report. (4) Refusals — no feature-map, dynamic-resolution or training API is exposed; a missing snapshot with `allow_download=False` raises `FileNotFoundError`; `evaluation_report` refuses to emit a metric without targets. No statistical mitigation (class re-balancing) is applied because the pipeline does not train.

###### Risks and harms

Overconfidence out of distribution: an unrelated image still yields a top-1 label with a score that can look high; the operator bears the harm when that label is acted on. Bias in the label space and training images: ImageNet's classes and their examples skew toward Western, web-scraped imagery, so objects common elsewhere are more often mislabelled; data subjects and third parties bear the harm when such labels feed downstream systems. Automation bias: reviewers presented with a confident label check less carefully. Silent preprocessing loss: the fixed 224-px input and the 0.9 center crop remove 10 % of each border and all fine detail of a large image, so a small or border-adjacent subject can be cropped or blurred away. Likelihood under normal photographic use is moderate for the first two and rises sharply off-distribution; magnitude ranges from a wrong tag to a wrongly moderated image.

###### Use cases

The pipeline must not be used for surveillance, biometric or demographic profiling, or social scoring — its label space cannot do these, and adapting it to try would be a misuse. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, nor deceptive or manipulative applications such as fabricating evidence of what an image contains. Any use that violates the Apache-2.0 terms of the upstream weights or the DIMER deployment terms is prohibited. The developers identify no further prohibited use beyond these because the model's output is a coarse object label.

## Immutable provenance

- Model: `timm/vit_base_patch16_224.orig_in21k_ft_in1k`
- Revision: `e0bd370de6799e8d1f47a911174ff4c3708e2323`
- Snapshot manifest: `weights/vit-base-p16-224/dimer-base-manifest.json`, `totalBytes` 346288708
- `model.safetensors` SHA-256: `669b949ea91fd19217f200cee259780bde32210c1eb9a5af3859f0dd8346b2ec` (346284714 bytes)
- `config.json` SHA-256: `91aa54e1244ba735215d4ec117b62c1a00a9f4a744b0e0543b16b901eaf73784` (585 bytes)
- Weight format: SafeTensors; loader `timm.create_model("vit_base_patch16_224", pretrained=True, pretrained_cfg_overlay={"file": ...})`; only the SafeTensors weight is staged and listed, so no pickle-format file is ever read

## Input/output contract

- `ViTClassificationPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`
- `predict(images, top_k=5)` — `images`: one `PIL.Image.Image` or a sequence of 1–64; sides 1–4096 px; any mode (converted to RGB). Returns `{"predictions": [{"predicted_index", "predicted_label", "top_k": [{"label", "index", "score"}, ...]}, ...], "top_k", "decision_rule", "device", "source", "model_id", "model_revision"}`; `score` is the softmax over 1000 classes.
- `validate_inputs(images, top_k=5, *, names=None)` — the same checks as `predict`, returned as an input manifest (`INPUT_SCHEMA`, per-input mode/size, verdict `accepted`, empty `findings`); raises exactly what `predict` raises.
- `evaluation_report(result, targets=None, *, sample_kind="synthetic")` — `not-measurable` without targets; `sample-sanity` with `top_k_accuracy` at k=1 and k=5 when one ImageNet-1k index per prediction is supplied.
- `top_k_accuracy(predictions, targets, k=1)` — accepts the `predictions` list above or plain index lists.
- `verify_snapshot(path=None)` / `stage_missing_files(path=None, *, allow_download=False, downloader=None)` — manifest checks; raise `FileNotFoundError` / `ValueError`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `timm==1.0.29`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Precision: float32 on both CPU and CUDA; preprocessing resize 248 → center-crop 224 (`crop_pct 0.9`, bicubic), mean/std 0.5/0.5/0.5 from the snapshot `config.json`; fixed input size, CLS-token pooling.
- Measured (Windows venv, CPU, `CUDA_VISIBLE_DEVICES=-1`, `from_pretrained(device="cpu")`): device `cpu`, source `local-snapshot`, load 3.19 s, first predict 0.22 s, second predict 0.06 s, total 3.40 s; top-1 on a synthetic 256×256 gradient image `whistle` (index 902) at score 0.0238, top-5 all below 0.024 — meaningless by construction.
- Tests: `pytest -q -o addopts= tests` — 24 passed, offline, no weights required (13 pipeline, 6 role-helper, 5 notebook-parity).

## References

- Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021. https://arxiv.org/abs/2010.11929
- Original JAX weights and code: https://github.com/google-research/vision_transformer
- Wightman. PyTorch Image Models. https://github.com/huggingface/pytorch-image-models
- Deng et al. ImageNet: A large-scale hierarchical image database. CVPR 2009. https://doi.org/10.1109/CVPR.2009.5206848
- Upstream card: https://huggingface.co/timm/vit_base_patch16_224.orig_in21k_ft_in1k

"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E supervised-adaptation workflow: the pinned ViT-B/16 ImageNet-1k snapshot is
digest-verified and loaded, a digest-pinned real image corpus (CC0 iNaturalist bird photographs, three of the six
species outside the ImageNet-1k label space) is fetched, validated and split, the ImageNet-1k inference contract
is exercised, the frozen 768-d pre-logits are scored by k-NN and by a linear probe (the frozen policy) beside a
majority floor, a bounded unfreeze of the last blocks is trained and selected against the probe on validation (the
unfrozen policy), the held-out split is scored, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "vit_classification_pipeline",
    "repo_name": "vit-classification-pipeline",
    "stem": "vit_classification",
    "notebook_name": "vit_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned ViT-B/16 `orig_in21k_ft_in1k` snapshot (safetensors, 346 MB), fetches 180 digest-pinned CC0 iNaturalist photographs of six "
        "bird species from the iNaturalist open-data bucket (19 MB, no credential), validates them and draws 108 / 24 / 48 "
        "training, validation and test images by a seeded stratified split, classifies three test images through the "
        "ImageNet-1k inference contract with an input manifest and a rejection probe, scores the frozen 768-d pre-logits on the test split by "
        "a majority floor, a cosine 5-NN vote and a linear probe (the **frozen policy**), trains a bounded unfreeze of the "
        "last two transformer blocks with the head and selects between it and the probe by validation log-loss (the "
        "**unfrozen policy**), scores the held-out split with the selected model, prints predictions before and after, "
        "exports the head and any trained blocks as safetensors with a manifest, and reloads that artifact into a fresh "
        "pipeline to verify parity. The default path needs no repository clone, no DIMER worker or service, no credential, "
        "no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.2 §5). On CPU the whole path takes about two "
        "minutes of model time after the downloads; a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a zip or folder "
        "in the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one zip in Colab, then choose **Run after** from "
        "that cell (it first puts the model back to the pinned base) to supply your own labelled images — at least **12 "
        "photographs for two labels** (6 per label), 15 for three, and 4 per label from four labels up "
        "(`min_byod_records(n_labels)`) — as a `.zip` or folder holding `labels.csv` (columns `id`, `file`, `label`) beside the image files — images are "
        "decoded from the archive, never extracted to disk. They pass through the same validation, seeded stratified "
        "image-disjoint split, floors, frozen-policy probe, unfrozen-policy training and selection, held-out evaluation, "
        "prediction, artifact export and reload-parity cells as the iNaturalist sample. The expected schema and the ceilings "
        "are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "pipeline_class": "ViTClassificationPipeline",
    "weights_key": "vit-base-p16-224",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "timm"],
    "title": "ViT-B/16 orig_in21k_ft_in1k — DIMER E2E supervised adaptation tutorial: a new head beyond ImageNet-1k, linear probe vs bounded unfreeze (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/vit-classification-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/vit-classification-pipeline/blob/main/tutorials/vit_classification_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fvit__base__patch16__224.orig__in21k__ft__in1k-ffcc4d?style=flat",
            "https://huggingface.co/timm/vit_base_patch16_224.orig_in21k_ft_in1k",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-huggingface%2Fpytorch--image--models-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/huggingface/pytorch-image-models",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2010.11929-b31b1b.svg", "https://arxiv.org/abs/2010.11929"),
    ],
    "capability": "ImageNet-1k single-label image classification (argmax plus top-k softmax scores) and bounded supervised adaptation to a label space the checkpoint does not have — a new linear head on the frozen 768-d pre-logits with an optional unfreeze of the last transformer blocks — measured by held-out accuracy and macro-F1, using the pinned ViT-B/16 `orig_in21k_ft_in1k` weights",
    "intro": (
        "At inference the Vision Transformer splits one normalised 3×224×224 tensor into 196 patches of 16×16 pixels, "
        "prepends a class token, runs 12 self-attention blocks and reads 1000 logits from the class token in a single "
        "forward pass; `predict` applies a softmax and reports the argmax class plus the top-k classes with their "
        "scores. The input size is fixed at 224×224 by the checkpoint, so every image is resized and centre-cropped to "
        "it. The carried pipeline module adds snapshot verification, input validation, a fixed output contract and the "
        "`top_k_accuracy`, `validate_inputs` and `evaluation_report` helpers. **The label space is fixed to the 1000 "
        "ImageNet-1k classes:** an image whose subject is outside it still receives a label, with a score that says "
        "nothing about the mismatch.\n\n"
        "What this notebook adds to inference is **supervised adaptation to a label space the checkpoint does not "
        "have, under an explicit frozen-vs-unfrozen policy**. The dataset is real: 180 CC0-licensed, research-grade "
        "iNaturalist photographs of six common North American birds (30 per species, one per observer per species), "
        "chosen a priori and pinned by photo id, byte size and SHA-256, fetched from the iNaturalist open-data bucket at "
        "run time and refused on any mismatch; every record keeps its observation URL and observer login. ImageNet-1k "
        "names three of the six (goldfinch, house finch, junco) and has no class for the three sparrows — Section 5 "
        "shows what the ImageNet head does with them. The carried `metrics.py` scores predictions by **accuracy** and "
        "**macro-F1** with per-class recall and a confusion matrix; a **majority floor** and a **cosine 5-NN vote** over "
        "the frozen 768-d pre-logits (the class token after the final norm, the vector the ImageNet head reads) frame "
        "the numbers. The **frozen policy** trains only a new six-way linear head on those frozen pre-logits (a linear "
        "probe); the **unfrozen policy** continues from that probe by training the last transformer blocks with the "
        "new head end to end, and the epoch with the lowest validation log-loss — which may be the probe itself — is "
        "kept. The ImageNet head is never trained or exported; `predict` keeps it. The adaptation question is whether "
        "unfreezing buys anything over the probe on 108 training photographs. Nothing here is a quality claim about "
        "your images: it is one seeded split of one small corpus."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter, and wants to see how a pretrained image classifier is given a label space it does not have — a linear probe on frozen features versus a bounded unfreeze — and how to choose between them honestly. No prior experience with Vision Transformers or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. CPU is adequate (about two minutes of model time); a GPU is faster.\n\n**Input → Model → Output.**\n\n| | ImageNet-1k classification | Adapted classification | Bounded adaptation |\n|---|---|---|---|\n| Input | one or more images, any size up to 4,096 px, resized and centre-cropped to 224 × 224 | one photograph | labelled photographs (108 training and 24 validation in the sample), split by species |\n| Model | ViT-B/16: 196 patches + a class token through 12 transformer blocks, then the ImageNet-1k head | the same backbone with a new six-way linear head | frozen policy: the head on frozen 768-d pre-logits; unfrozen policy: also the last two blocks; validation log-loss chooses |\n| Output | argmax label and top-k softmax scores over 1000 classes — not calibrated | a species with its head probability | a safetensors adapter, and held-out accuracy and macro-F1 beside a majority floor and a 5-NN vote |\n\n**How to use this notebook.** Choose a runtime (CPU works; a GPU is faster), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package (including the 180-photograph table) and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer that names the run it quotes — the Kaggle T4 release run of 19 September 2026 or the build record. Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the corpus, validation and a stratified split *(evaluation practice)* → 5 the ImageNet-1k inference contract *(core concept: a fixed label space)* → 6 the floor, the 5-NN vote and the linear probe *(evaluation practice)* → 7 the unfrozen policy and validation selection *(core concept)* → 8 held-out evaluation → 9 predictions before and after, export and reload *(engineering)* → 10 change one thing (optional) → conclude."
            )
        ]
    },
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned real image corpus and validate and split it "
        "without leakage; classify images through the public ImageNet-1k API and read the argmax decision and the uncalibrated top-k scores correctly; read accuracy and "
        "macro-F1 beside a majority floor and a k-NN baseline; train a linear probe on the frozen pre-logits and a bounded "
        "unfreeze with explicit hyperparameters and validation-based selection between the two policies; evaluate on an "
        "independent test split; compare predictions before and after; and export a safetensors adapter (head plus any "
        "trained blocks) that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "object detection, segmentation, multi-label tagging, OCR, open-vocabulary or zero-shot classification, "
        "attention maps, data augmentation, full-backbone or patch-embedding training, any training of the ImageNet-1k "
        "head, any ImageNet-1k accuracy claim, and any claim that six bird species from one photo site stand in for "
        "your images. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with Vision Transformers or fine-tuning. The notebook explains pre-logits, the linear probe, the frozen and unfrozen policies, log-loss, macro-F1 and the adapter where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it. The default path runs on CPU and uses CUDA automatically when available; float32 on both. Each 224 px image costs about 16.9 GMACs (upstream README): the build record measured about 0.04 s per image on CPU (6.4 s for the 180-image k-NN pass), a 7 s linear probe including feature extraction, and about 6 s per unfreeze epoch over 108 images plus a 24-image validation pass. The pinned `torch==2.14.0` install and the 346 MB checkpoint are the large downloads of the run, then the 19 MB of photographs.",
        "- **Knowledge:** basic Python and NumPy; what a softmax over class logits is; what a linear probe is and why it is the cheapest honest test of a representation; what accuracy and macro-F1 measure and why macro-F1 punishes a forgotten class; what validation-based selection between two policies means.",
        "- **Data contract:** records are `{id, image, label}` — a PIL image (or a path to one) with each side 1..4,096 px (the pipeline's own ceiling; images are resized and centre-cropped to 224 px, never rejected for being small), a label of 1..64 plain characters, ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; a training split needs 8..20,000 records and 2..100 classes, and every label needs one training, one validation and one test photograph, so with the default 15 % + 20 % stratified hold-out the effective BYOD minimum is **12 photographs for two labels** (6 per label), 15 for three, and 4 per label from four labels up (`min_byod_records(n_labels)`); images are de-duplicated by decoded-pixel digest before splitting so the same photograph never sits in two splits. BYOD accepts a `.zip` (or a directory) holding `labels.csv` and the image files.",
        "- **Validation is structural, not semantic:** nothing checks that a label is right for its image — a mislabelled set is trained on without complaint; observers can contribute to more than one split (the notebook counts them) because the sample is stratified by species, not by observer.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of people or private places are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 180 pinned objects (`<photo id>/medium.jpg` or `medium.jpeg`, each photo's own served extension recorded in the table, 19,183,071 bytes in total, one SHA-256 each in the carried `SAMPLE_RECORDS` table) from `inaturalist-open-data.s3.amazonaws.com` over HTTPS, each refused on any byte-size or SHA-256 mismatch before it is decoded; every photograph is CC0 by its own iNaturalist licence code and credited to its observer in the records.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and split\n\n"
                "`fetch_corpus` downloads the 180 pinned photographs (or reads them from the cache), refuses a byte-size "
                "or SHA-256 mismatch per file before it is decoded, and `read_corpus` turns each into a `{{id, image, "
                "label}}` record with its species names, observation URL and observer. `build_sample_dataset` draws 18 "
                "training, 4 validation and 8 test photographs per species by a seeded stratified shuffle; "
                "`validate_dataset` then checks every record against the contract, `check_split_disjoint` asserts no "
                "photograph (by decoded-pixel digest) appears in two splits, `observer_overlap` counts the observers who "
                "contributed to more than one split — an observation, since the sample is stratified by species — and "
                "the training labels table is written to `outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 180 photographs, six classes of 30, splits 108 / 24 / 48, three digests, and four refusal "
                "probes — a duplicate id, an oversized image, a single-class dataset and a dataset too small to split — "
                "each rejected before `torch` does anything. About a minute on the first run for the downloads.\n\n"
                "*Evaluation practice.* **Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a zip or "
                "a folder holding `labels.csv` and the images, as a path in this runtime — this works on Colab, Kaggle and "
                "Jupyter) or leave `BYOD_PATH` empty to upload exactly one zip through the Colab dialog; then choose **Run after** "
                "from this cell. This cell first puts the model back to the pinned base, so the 5-NN vote and the probe read the "
                "untouched backbone. The effective minimum is 12 photographs for two labels.\n\n"
                "**Predict before running:** the split is stratified by species, not by observer. Will any observer's photographs "
                "land in more than one split, and why might that matter?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import time\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "# A re-run after Sections 6-7 (BYOD, or a new split): the 5-NN vote and the probe must read the pinned base.\n"
                "had_adapter = pipe.adapter is not None\n"
                "restored_blocks = pipe.restore_base()\n"
                "if had_adapter or restored_blocks:\n"
                "    print({{'restored_pinned_base': len(restored_blocks), 'note': 'the head and any trained blocks were removed; Sections 5-7 start from the checkpoint again'}})\n"
                "t0 = time.perf_counter()\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding labels.csv and the images.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding labels.csv and the images.')\n"
                "        byod_path = Path('work') / 'byod.zip'\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    n_labels = len({{r['label'] for r in records}})\n"
                "    raw_count = {{'byod': len(records), 'labels': n_labels, 'duplicate_images_dropped': len(records) - sum(len(part) for part in splits.values()), 'effective_minimum': min_byod_records(max(n_labels, 2))['total']}}\n"
                "    if len(splits['test']) < 5 * n_labels:\n"
                "        print({{'caution': f\"only {{len(splits['test'])}} held-out test photographs for {{n_labels}} labels (fewer than 5 per label): accuracy moves in large steps and carries no dispersion estimate; add photographs before reading it\"}})\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/inat-birds'))\n"
                "    raw_count = {{'photographs': len(corpus), 'species': len(SPECIES)}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "fetch_seconds = round(time.perf_counter() - t0, 1)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "# The training split must hold MIN_RECORDS; validation and test only need one photograph each (split_dataset checks the labels).\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "classes = class_names(train_records)\n"
                "disjoint = check_split_disjoint(splits)\n"
                "overlap = observer_overlap(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw': raw_count, 'splits': disjoint, 'classes': classes, 'observer_overlap': overlap, 'fetch_seconds': fetch_seconds, 'corpus_bytes': CORPUS_BYTES}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'label_counts': manifest['label_counts'], 'image_side': manifest['image_side'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{k: example[k] for k in ('id', 'label', 'observer', 'inat_observation_url') if k in example}}, 'size': example['image'].size}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'oversized image': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'single class': [{{**r, 'label': 'bird'}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** 180 photographs, 108 / 24 / 48, the three digests, `observer_overlap`, and the four refusals.\n\n<details><summary>Check your reasoning</summary>Yes: the sample keeps one photograph per observer per species, but an observer who photographed two species can land in two splits — a review probe counted 31 of 117 observers in more than one split with the default seed. A head can then learn a photographer's camera or background rather than the bird, so the held-out score may be optimistic. Splitting by observer would remove that; the notebook reports the overlap instead of hiding it.</details>"
            ),
        },
        {
            "md": (
                "## 5. Classify through the ImageNet-1k inference contract\n\n"
                "Before any adaptation, the inference contract is exercised as it always was, on three test "
                "photographs. `validate_inputs` applies exactly the checks `predict` applies — type, batch size "
                "1..`MAX_BATCH`, image side 1..`MAX_IMAGE_SIDE` px, `top_k` 1..`NUM_CLASSES` — and returns an input "
                "manifest; a deliberately oversized image is validated too and its rejection recorded as a finding. "
                "`predict` returns, per image, `predicted_index` / `predicted_label` and a `top_k` list ordered by "
                "descending softmax score under the `argmax` decision rule; the score is a softmax over uncalibrated "
                "logits, not a probability of being right. Read the three top-5 lists against the gold species: a "
                "goldfinch, a house finch or a junco can be named, because ImageNet-1k has those classes (indices 11, "
                "12, 13); a chipping, song or white-throated sparrow cannot, and the head still answers — the build "
                "record saw a chipping sparrow labelled `ruffed grouse` at 0.26 and a song sparrow `brambling` at "
                "0.37. That label-space gap, not a quality defect, is what the rest of the notebook adapts around. The "
                "frozen predictions that Section 9 compares against come later, from the probe.\n\n"
                "*Core concept.* A classifier's label space is fixed by its head: it always answers with one of its own "
                "classes, and its softmax score is not a confidence that the answer is right.\n\n"
                "**Predict before running:** ImageNet-1k has no class for a chipping sparrow. What will the head say for one, "
                "and with what score?"
            ),
            "code": (
                "probe_records = test_records[:3]\n"
                "print({{'ceilings': {{'NUM_CLASSES': NUM_CLASSES, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_BATCH': MAX_BATCH}}, 'contract': {{'INPUT_SIZE': INPUT_SIZE, 'FEATURE_DIM': FEATURE_DIM, 'DECISION_RULE': DECISION_RULE, 'TRANSFORMER_BLOCKS': TRANSFORMER_BLOCKS, 'PARAMETER_COUNT': PARAMETER_COUNT}}}})\n"
                "input_manifest = validate_inputs([r['image'] for r in probe_records], top_k=5, names=[r['id'] for r in probe_records])\n"
                "try:\n"
                "    validate_inputs(Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8)))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversized-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "started = time.perf_counter()\n"
                "imagenet_result = pipe.predict([r['image'] for r in probe_records], top_k=5)\n"
                "predict_seconds = round(time.perf_counter() - started, 3)\n"
                "checks = {{\n"
                "    'one_prediction_per_image': len(imagenet_result['predictions']) == 3 and imagenet_result['top_k'] == 5,\n"
                "    'ranked_descending': all(all(a['score'] >= b['score'] for a, b in zip(p['top_k'], p['top_k'][1:])) for p in imagenet_result['predictions']),\n"
                "    'argmax_is_rank_one': all(p['predicted_index'] == p['top_k'][0]['index'] for p in imagenet_result['predictions']),\n"
                "    'scores_in_unit_interval': all(0.0 <= item['score'] <= 1.0 for p in imagenet_result['predictions'] for item in p['top_k']),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'predict output failed a sanity check: {{checks}}')\n"
                "print({{'probe_ids': [r['id'] for r in probe_records], 'decision_rule': imagenet_result['decision_rule'], 'seconds': predict_seconds, 'device': imagenet_result['device'], 'checks': checks, 'findings': len(input_manifest['findings'])}})\n"
                "for record, prediction in zip(probe_records, imagenet_result['predictions']):\n"
                "    print({{'id': record['id'], 'gold_species': record['label'], 'imagenet_top5': [(item['index'], item['label'].split(',')[0], round(item['score'], 3)) for item in prediction['top_k']]}})\n"
                "print({{'note': 'ImageNet-1k names goldfinch (11), house finch (12) and junco (13); it has no class for the three sparrows, and the head answers anyway'}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the gold species beside each top-5 list, and the score of the top label for a sparrow.\n\n<details><summary>Check your reasoning</summary>Some other ImageNet class, with an unremarkable score: the build record saw a chipping sparrow labelled `ruffed grouse` at 0.26 and a song sparrow `brambling` at 0.37. Nothing in the output flags the mismatch, which is why the rest of the notebook trains a new head instead of reading the old one.</details>'
            ),
        },
        {
            "md": (
                "## 6. The majority floor, the k-NN baseline and the frozen policy\n\n"
                "Three numbers frame the adaptation, all on the 48 test photographs. The **majority floor** predicts the "
                "most frequent training species for every image — 1 / 6 here, since the sample is balanced. The **cosine "
                "5-NN vote** labels each test image by its five nearest training images in the frozen feature space: what "
                "the representation gives with no training at all. The **frozen policy** is `pipe.adapt` with "
                "`trainable_blocks=0`: a new six-way linear head over the frozen, L2-normalised 768-d pre-logits (`pipe.features`; the ImageNet head is bypassed, not trained), trained "
                "full-batch with AdamW for `PROBE_STEPS` steps — the linear probe — scored on validation as epoch 0 of its "
                "history and then on the test split by `pipe.evaluate` (accuracy, macro-F1, per-class recall, "
                "confusion). **What to look for:** the probe against the 5-NN vote — how much a trained head adds over the raw "
                "feature geometry — and the per-class recall. About fifteen seconds on CPU.\n\n"
                "*Evaluation practice.* The cell reports a **verdict** — whether the probe beats the majority floor — and records "
                "it; on a small BYOD test split a miss is a finding, and the notebook continues. The policy identity stays a "
                "hard check: `trainable_blocks=0` must give the frozen policy.\n\n"
                "**Predict before running:** the 5-NN vote uses no training at all. How close will it come to the trained "
                "probe, and which species will both find hardest?"
            ),
            "code": (
                "PROBE_STEPS = 300  # @param {{type:\"integer\"}}\n"
                "PROBE_LR = 0.01  # @param {{type:\"number\"}}\n\n"
                "def brief(m):\n"
                "    return {{'accuracy': round(m['accuracy'], 4), 'macro_f1': round(m['macro_f1'], 4), 'n': m['n']}}\n\n"
                "floor = majority_baseline([r['label'] for r in train_records], [r['label'] for r in test_records], classes)\n"
                "print({{'majority_floor': brief(floor), 'baseline': floor['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "baseline_knn = pipe.knn_baseline(train_records, test_records, k=5)\n"
                "print({{'knn_baseline': brief(baseline_knn), 'baseline': baseline_knn['baseline'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "probe_result = pipe.adapt(train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=0)\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_policy': probe_result['policy'], 'probe_final_loss': round(probe_result['probe_final_loss'], 4), 'validation': {{k: round(v, 4) for k, v in probe_result['history'][0]['val'].items()}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'frozen_policy_test': brief(frozen_test), 'log_loss': round(frozen_test['log_loss'], 4), 'verdict': frozen_test['verdict'], 'per_class_recall': {{c: round(v['recall'], 2) for c, v in frozen_test['per_class'].items()}}}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "if not probe_result['policy'].startswith('frozen'):  # a contract check: trainable_blocks=0 is the frozen policy by definition\n"
                "    raise RuntimeError(f\"adapt(trainable_blocks=0) returned policy {{probe_result['policy']!r}}\")\n"
                "# A reported verdict, not an assertion: on a small BYOD test split the probe may miss the floor, and that is a finding.\n"
                "frozen_verdict = 'above floor' if frozen_test['accuracy'] > floor['accuracy'] else 'at or below floor'\n"
                "print({{'frozen_vs_floor': frozen_verdict}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the floor, the 5-NN vote and the probe on accuracy and macro-F1, and the per-class recall.\n\n<details><summary>Check your reasoning</summary>Close. In the Kaggle T4 release run (19 September 2026) the floor scored 16.7 %, the 5-NN vote 75.0 % (macro-F1 0.752) and the probe 79.2 % (0.789): the frozen features already separate the six species, and a trained head adds about two photographs of 48. The song sparrow was the weak class for both, at a recall of 0.5; read the confusion matrix in Section 8 to see where its misses go.</details>'
            ),
        },
        {
            "md": (
                "## 7. The unfrozen policy: a bounded unfreeze selected against the probe\n\n"
                "`pipe.adapt` with `TRAINABLE_BLOCKS` > 0 first retrains the linear probe on the frozen features (epoch 0 "
                "of the history, the frozen policy), then unfreezes the last `TRAINABLE_BLOCKS` transformer blocks — "
                "two by default, 14,175,744 of 86,567,656 parameters; the patch embedding, the position embedding, "
                "the earlier blocks, the final norm and the ImageNet head stay frozen — and trains them with the new head end to end on the "
                "photographs for `EPOCHS` epochs (AdamW at `LEARNING_RATE`, weight decay 0.01, gradient clipping 1.0, "
                "seeded shuffling, no augmentation). Every epoch is scored on validation, and the epoch with the "
                "**lowest validation log-loss** is kept — epoch 0, the probe, competes on equal terms, so the selected "
                "policy can be either. Accuracy and macro-F1 are printed beside the loss at every epoch.\n\n"
                "Watch the validation loss: with 24 validation photographs the probe is already near-perfect, so the "
                "unfreeze must beat it on confidence, not just on the label. The build record's sweep on this sample: two blocks at 3e-5 for four epochs never beat the probe's validation log-loss (0.162 against 0.163–0.166), so **the probe was selected**; two blocks at 1e-4 was not selected either; four blocks at 1e-4 and two blocks at 3e-4 were selected on validation and lost 6.3 and 12.5 points on the held-out split.\n\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: blocks an earlier call (or an "
                "artifact) changed are restored before the probe's features are computed, so epoch 0 is always a probe on the "
                "frozen backbone and re-running Sections 6–8 with a changed field repeats the comparison validly. To compare a "
                "change side by side without replacing the default exports, use Section 10.\n\n"
                "**Predict before running:** with 24 validation photographs and a probe already near-perfect on them, will the "
                "unfreeze beat the probe's validation log-loss?"
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 3e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_BLOCKS = 2  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'stage': entry['stage'], 'train_loss': round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_log_loss'] = round(entry['val']['log_loss'], 4)\n"
                "        row['val_accuracy'] = round(entry['val']['accuracy'], 4)\n"
                "        row['val_macro_f1'] = round(entry['val']['macro_f1'], 4)\n"
                "    print(row)\n\n"
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'trainable_blocks': TRAINABLE_BLOCKS, 'probe_steps': PROBE_STEPS, 'probe_lr': PROBE_LR}}\n"
                "if settings != {{'epochs': 4, 'lr': 3e-5, 'batch_size': 8, 'trainable_blocks': 2, 'probe_steps': 300, 'probe_lr': 0.01}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9; Section 10 compares a change side by side instead', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=TRAINABLE_BLOCKS, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'selected_policy': adapt_result['policy'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'trainable_head': adapt_result['n_trainable_head'], 'trainable_blocks': adapt_result['n_trainable_blocks'], 'total_parameters': adapt_result['n_total'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the epoch-0 validation log-loss against each later epoch's, and `selected_policy`.\n\n<details><summary>Check your reasoning</summary>No — in the release run validation kept the probe (`frozen backbone + linear probe`), as in the build record, where the probe's 0.162 was never beaten (0.163–0.166). With so few validation photographs the probe already labels them right, so the unfreeze can only win on confidence, and four epochs at 3e-5 were not enough.</details>"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or policy selection, and no photograph in it appears in the "
                "training or validation splits. The selected model is scored exactly as the frozen policy was in Section 6, "
                "and the four rows are put side by side: majority floor, k-NN vote, frozen policy, selected policy. Read "
                "the policy first: if validation kept the probe, the last two rows are the same model; if it chose the "
                "unfreeze, the delta is what the unfreeze bought on 48 photographs — the build record's validation kept "
                "the probe at the default settings, and the two unfreezes it did select lost accuracy on the held-out split. The "
                "cell records verdicts — whether the selected model beats the majority floor, and whether it improved on the "
                "probe — instead of asserting them, so a small BYOD set at the floor still exports, reloads and writes the "
                "result. 48 photographs from one seeded split of one corpus give no dispersion estimate — one image is about "
                "two points of accuracy.\n\n"
                "**Predict before running:** if validation kept the probe, what will `delta_vs_frozen` be?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{\n"
                "    metric: {{'majority_floor': round(floor[metric], 4), 'knn5': round(baseline_knn[metric], 4), 'frozen_policy': round(frozen_test[metric], 4), 'selected_policy': round(adapted_test[metric], 4)}}\n"
                "    for metric in ('accuracy', 'macro_f1')\n"
                "}}\n"
                "comparison['log_loss'] = {{'frozen_policy': round(frozen_test['log_loss'], 4), 'selected_policy': round(adapted_test['log_loss'], 4)}}\n"
                "comparison['per_class_recall'] = {{c: {{'knn5': round(baseline_knn['per_class'][c]['recall'], 2), 'frozen': round(frozen_test['per_class'][c]['recall'], 2), 'selected': round(adapted_test['per_class'][c]['recall'], 2)}} for c in classes}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 4) for metric in ('accuracy', 'macro_f1')}}\n"
                "comparison['selected_policy'] = adapt_result['policy']\n"
                "# Reported verdicts, not assertions: a result at the floor is a finding, and export and reload still run.\n"
                "comparison['verdicts'] = {{\n"
                "    'frozen_vs_floor': frozen_verdict,\n"
                "    'selected_vs_floor': 'above floor' if adapted_test['accuracy'] > floor['accuracy'] else 'at or below floor',\n"
                "    'selected_vs_frozen_accuracy': 'improved' if adapted_test['accuracy'] > frozen_test['accuracy'] else ('no gain' if adapted_test['accuracy'] == frozen_test['accuracy'] else 'worse'),\n"
                "}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "print({{'confusion_selected': adapted_test['confusion'], 'classes': classes}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'observer_overlap': overlap,\n"
                "    'classes': classes,\n"
                "    'baselines': {{'majority_floor': floor, 'knn5': baseline_knn}},\n"
                "    'frozen_policy': {{'adaptation': {{k: v for k, v in probe_result.items() if k not in ('history', 'trainable_names')}}, 'history': probe_result['history'], 'test': frozen_test}},\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'verdicts': comparison['verdicts']}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** `selected_policy`, `delta_vs_frozen`, the per-class recall and the `verdicts`.\n\n<details><summary>Check your reasoning</summary>Zero: the last two rows are then the same model. In the release run validation kept the probe, so both scored 79.2 % accuracy and 0.789 macro-F1 and `delta_vs_frozen` was 0. The build record's sweep is the warning: the two unfreezes validation did select lost 6.3 and 12.5 points on the held-out split — a validation choice on 24 photographs is itself noisy.</details>"
            ),
        },
        {
            "md": (
                "## 9. Predict before and after, export the adapter and reload it\n\n"
                "Six test photographs are labelled by `pipe.classify` with the selected model and printed beside the "
                "frozen policy's predictions (recomputed from the probe's stored head on the frozen features — the head "
                "and blocks of the frozen policy were saved before the unfreeze) and the gold species with the top "
                "probability; read the probabilities as the head's softmax, not a calibrated confidence. The per-batch "
                "`evaluation_report` helper — the inference-stage helper for `predict` — is written for the three probe "
                "images and stays `not-measurable`, because the species labels are not ImageNet-1k indices; "
                "`pipe.evaluate` is that labelled evaluation.\n\n"
                "`pipe.save_artifact` writes the head (`head.weight`, `head.bias`) and, when the unfrozen policy was "
                "selected, the trained block tensors — 18.6 KB for the probe alone, about 56.7 MB with two trained blocks — as `adapter.safetensors`, with a `manifest.json` "
                "recording the artifact format, the base model id and revision, the digest of the base `model.safetensors`, "
                "the classes, the selected policy, the tensor names, the file size and SHA-256, the training configuration "
                "and the epoch history (OUT8). `ViTClassificationPipeline.from_artifact` re-verifies the base "
                "snapshot, checks the artifact manifest and digest **before** deserialising, rebuilds the head from the "
                "manifest's classes, refuses any tensor that is not a transformer-block tensor of the base, and overlays "
                "the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The "
                "ImageNet head is never in the artifact. The cell asserts identical probabilities on the six images, an identical test accuracy, and identical `predict` output from the reloaded pipeline (VER4) — a contract check, so it stays a hard check.\n\n"
                "**Predict before running:** if the probe was selected, how many of the six predictions will change between "
                "the frozen and the selected column?"
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "show = test_records[:6]\n"
                "after = pipe.classify([r['image'] for r in show])\n"
                "frozen_pipe = ViTClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "frozen_pipe.adapt(train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=0)\n"
                "before = frozen_pipe.classify([r['image'] for r in show])\n"
                "rows = []\n"
                "for i, record in enumerate(show):\n"
                "    rows.append({{'id': record['id'], 'gold': record['label'], 'frozen_label': before['labels'][i], 'frozen_top_probability': round(max(before['probabilities'][i]), 4), 'selected_label': after['labels'][i], 'selected_top_probability': round(max(after['probabilities'][i]), 4), 'observer': record.get('observer', ''), 'inat_observation_url': record.get('inat_observation_url', '')}})\n"
                "    print({{k: rows[-1][k] for k in ('id', 'gold', 'frozen_label', 'frozen_top_probability', 'selected_label', 'selected_top_probability')}})\n"
                "single_report = evaluation_report(imagenet_result, sample_kind='three iNaturalist test photographs' if not USE_BYOD else 'three BYOD test images')\n"
                "print({{'batch_report_verdict': single_report['verdict'], 'selected_policy': after['policy'], 'predictions_changed': sum(r['frozen_label'] != r['selected_label'] for r in rows), 'of': len(rows)}})\n"
                "with open('outputs/{stem}_predictions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(rows)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'policy': artifact_manifest['adapter']['policy'], 'classes': artifact_manifest['adapter']['classes'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = ViTClassificationPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_after = reloaded.classify([r['image'] for r in show])\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "parity = {{'probabilities_identical': reloaded_after['probabilities'] == after['probabilities'], 'accuracy_in_memory': round(adapted_test['accuracy'], 6), 'accuracy_reloaded': round(reloaded_test['accuracy'], 6), 'classes_identical': reloaded.classes == pipe.classes, 'predict_identical': reloaded.predict([r['image'] for r in probe_records], top_k=5)['predictions'] == pipe.predict([r['image'] for r in probe_records], top_k=5)['predictions']}}\n"
                "print({{'reload_parity': parity, 'reloaded_policy': reloaded.adapter['policy'], 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['probabilities_identical'] and parity['classes_identical'] and parity['predict_identical'] and abs(adapted_test['accuracy'] - reloaded_test['accuracy']) < 1e-9\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHTS_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'photographs': len(SAMPLE_RECORDS), 'bytes': CORPUS_BYTES, 'license': CORPUS_LICENSE, 'species': SPECIES}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'probe_ids': [r['id'] for r in probe_records], 'imagenet_top1': [p['predicted_label'] for p in imagenet_result['predictions']], 'seconds': predict_seconds}},\n"
                "    'comparison': comparison,\n"
                "    'predictions_before_after': rows,\n"
                "    'batch_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors']), 'policy': artifact_manifest['adapter']['policy']}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                '**What to notice:** `predictions_changed`, the two probability columns, and the reload parity line.\n\n<details><summary>Check your reasoning</summary>None, if the probe was selected: both columns come from a probe trained the same way on the same frozen features. Reload parity held in the release run: identical probabilities, identical `predict` output, and test accuracy 0.7917 in memory and reloaded.</details>'
            ),
        },
        {
            "md": (
                "## 10. Change one thing: four blocks at a higher rate (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** setting — by default four blocks at 1e-4 "
                "instead of two at 3e-5 — and run this cell after Sections 4–9. The experiment loads its **own** pipeline from the "
                "verified snapshot, so it starts from the checkpoint and never touches the default `pipe`; it writes only to "
                "`outputs/{stem}_experiment/`, prints the default and the changed run side by side (including the epoch-0 "
                "validation log-loss, which must match), and checks that the default exports are byte-identical afterwards.\n\n"
                "**Predict:** will validation select the unfreeze this time, and will the held-out accuracy agree with it?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_TRAINABLE_BLOCKS = 4  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshot: the experiment starts from the checkpoint and the default pipe is untouched.\n"
                "    experiment_pipe = ViTClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=EXPERIMENT_TRAINABLE_BLOCKS, epochs=EXPERIMENT_EPOCHS, lr=EXPERIMENT_LEARNING_RATE, batch_size=BATCH_SIZE, progress=report)\n"
                "    experiment_test = experiment_pipe.evaluate(test_records)\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'trainable_blocks': adapt_result['trainable_blocks'], 'epochs': adapt_result['epochs'], 'lr': adapt_result['lr']}}, 'experiment': {{'trainable_blocks': EXPERIMENT_TRAINABLE_BLOCKS, 'epochs': EXPERIMENT_EPOCHS, 'lr': EXPERIMENT_LEARNING_RATE}}}},\n"
                "        'selected_policy': {{'default': adapt_result['policy'], 'experiment': experiment_result['policy']}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'validation_log_loss_by_epoch': {{'default': [round(e['val']['log_loss'], 4) for e in adapt_result['history'] if e.get('val')], 'experiment': [round(e['val']['log_loss'], 4) for e in experiment_result['history'] if e.get('val')]}},\n"
                "        'test': {{metric: {{'frozen_policy': round(frozen_test[metric], 4), 'default': round(adapted_test[metric], 4), 'experiment': round(experiment_test[metric], 4)}} for metric in ('accuracy', 'macro_f1')}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history']}}, handle, indent=2, ensure_ascii=False, default=str)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare the two validation-loss curves (epoch 0 must be equal) and the `test` rows.\n\n"
                "<details><summary>Check your reasoning</summary>In the build record four blocks at 1e-4 were selected on "
                "validation at epoch 4 and scored 72.9 % on the test split — 6.3 points below the probe. A lower validation "
                "loss on 24 photographs did not carry over to 48 held-out ones: more capacity began to fit the training "
                "photographs. No experiment run is recorded on the release runtime; read a difference of one or two photographs "
                "as noise.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen ViT-B/16 pre-logits — trained to separate 1000 ImageNet classes, three of which are these birds "
        "— already separate all six species well enough that a cosine 5-NN vote reaches 75 % and a linear probe 79 % "
        "on 48 held-out photographs against a majority floor of one in six, and a bounded unfreeze of the last two "
        "blocks on 108 training photographs never beat the probe's validation log-loss, so validation kept the probe. "
        "That is the claim and the finding: the adaptation contract gives the checkpoint a label space it did not "
        "have, runs both policies end to end on a real labelled corpus, chooses between them on validation rather "
        "than by assumption, and reports the answer against a floor and a no-training baseline rather than in "
        "isolation.\n\n"
        "The test split is 48 photographs from one seeded split of one small corpus with no dispersion estimate — one "
        "image is about two points of accuracy, so a two-point delta is noise. Observers contribute to more than one "
        "split (the notebook counts them), so some of what the head learns may be a photographer's style rather than a "
        "bird. Accuracy and macro-F1 say whether the gold species is predicted, not whether the features are good for "
        "any other task; the head's softmax is not a calibrated confidence. When the unfrozen policy is selected it "
        "changes the last blocks, which every input shares, so `predict` — which keeps the untouched ImageNet head — "
        "returns different ImageNet-1k outputs after it; the artifact records which policy won.\n\n"
        "Three things to carry to real data. **Floors first:** the majority floor and the k-NN vote on *your* images are "
        "the numbers to read before any trained head's — if the probe barely beats k-NN, the features already carry the "
        "task. **Leakage:** split by photographer, session or device when your images come from one (the contract "
        "de-duplicates by pixels, not by source). **Policy:** unfreezing is a hypothesis to test on validation, not a "
        "default; with a few hundred images the probe is usually the honest choice — here it was — and the two "
        "unfreezes that validation did select in the build sweep lost six and twelve points on the held-out split.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled image "
        "corpus, validate the demonstrated dataset contract without leakage, execute the ImageNet-1k inference contract, "
        "a linear probe and a bounded unfreeze with validation-based policy selection, evaluate by accuracy and macro-F1 "
        "against a floor and a k-NN baseline on an independent split, and emit the shown machine-readable artifacts — "
        "without the repository being reachable. It does **not** establish benchmark superiority, ImageNet-1k accuracy, "
        "a usable acceptance threshold, or production fitness.\n\n"
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 trains four blocks at "
        "1e-4 in its own pipeline and prints it beside the default run — change `EXPERIMENT_TRAINABLE_BLOCKS`, "
        "`EXPERIMENT_EPOCHS` or `EXPERIMENT_LEARNING_RATE` there (the build record's 3e-4 was selected at epoch 2 and scored "
        "66.7 %) and run that cell again. Changing `EPOCHS`, `LEARNING_RATE` or `TRAINABLE_BLOCKS` and choosing **Run after** "
        "from Section 7 also starts from the pinned base — every `adapt` restores it first — but replaces the default "
        "results and exports. BYOD: `USE_BYOD` and `BYOD_PATH` in Section 4, then **Run after** from Section 4, and read "
        "the k-NN baseline before either policy.\n\n"
        "## Troubleshooting\n\n"
'- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering `files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and choose **Run all**; leave the optional experiment off on a small runtime.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again; the snapshot comes from `huggingface.co`.\n- **Section 4 cannot fetch a photograph, or one fails its digest** — `fetch_corpus` names it; the default path needs `inaturalist-open-data.s3.amazonaws.com`. Run Section 4 again (cached photographs are re-hashed); delete `weights/inat-birds/` if a cached file is corrupt.\n- **Out of memory** — lower `BATCH_SIZE` in Section 7, or restart the session and choose **Run all**; leave Section 10 off on a small runtime.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip in the runtime (or attach it as a dataset) and set `BYOD_PATH`.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **BYOD: "labels.csv line N (file …): that image file is not in the dataset"** — fix the `file` column of that row, or add the image to the zip.\n- **BYOD: "… Pillow cannot decode the image"** — the file on that line is corrupt or not an image.\n- **BYOD: "split leaves no train/validation photograph for labels …" or "… training records"** — add photographs; the message names the minimum for your label count.\n'
        "## Glossary\n\n"
        "- **Pre-logits** — the 768-d class-token vector after the final norm, which the ImageNet head reads.\n"
        "- **Class token** — a learned token prepended to the 196 patches whose output summarises the image.\n"
        "- **Linear probe** — a new linear head trained on frozen features: the cheapest honest test of a representation.\n"
        "- **Frozen / unfrozen policy** — the probe alone; the probe plus the last transformer blocks trained end to end.\n"
        "- **Log-loss** — the mean negative log-probability of the gold label; it rewards confidence as well as "
        "correctness.\n"
        "- **Accuracy / macro-F1** — the fraction right; the mean of per-class F1, which punishes a forgotten class.\n"
        "- **Majority floor / 5-NN vote** — the most frequent training label for everything; the five nearest training "
        "photographs in feature space.\n"
        "- **Validation selection** — choosing the policy and epoch by validation log-loss, never by the test split.\n"
        "- **Observer overlap** — photographers whose images sit in more than one split.\n"
        "- **Adapter / reload parity** — the trained tensors only (safetensors) overlaid on the pinned base; the reloaded "
        "pipeline gives identical outputs.\n"
        "- **BYOD** — bring your own data: your labelled photographs through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The task was ___ labels on ___ test photographs.\n"
        "- The floor scored ___, the 5-NN vote ___ and the probe ___ (macro-F1 ___).\n"
        "- Validation selected the ___ policy at epoch ___; on the test split it scored ___, so against the probe it ___.\n"
        "- The weakest class was ___ (recall ___).\n"
        "- What I would need before claiming the unfreeze helps: ___ (for example more test photographs, several seeds).\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/vit-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/vit-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/vit-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/huggingface/pytorch-image-models\n"
        "- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2020): https://arxiv.org/abs/2010.11929\n"
        "- iNaturalist open data (CC0 photographs credited to their observers in the carried records): https://www.inaturalist.org/pages/developers\n"
        "- timm documentation: https://huggingface.co/docs/timm\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}

"""ImageNet-1k classification with the pinned ``timm/vit_base_patch16_224.orig_in21k_ft_in1k`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/<key>/``) or,
when explicitly allowed, from the Hugging Face Hub at the pinned revision. Preprocessing is
the upstream ``pretrained_cfg`` (resize/crop/normalize) resolved through ``timm.data``: the
ViT-B/16 input is fixed at 3x224x224, so every image is resized and center-cropped to that size.

The adaptation contract (``features``, ``knn_baseline``, ``adapt``, ``evaluate``, ``classify``,
``save_artifact``, ``from_artifact``) trains a new linear head on the frozen 768-d pre-logits of a validated
``{id, image, label}`` dataset (the **frozen policy**), optionally continues with a bounded unfreeze of the
last transformer blocks (the **unfrozen policy**), scores held-out images by accuracy and macro-F1, and
exports the new head plus any trained blocks as a safetensors adapter bound to the pinned base weights. The
ImageNet-1k head and ``predict`` are unchanged by it, but ``predict`` reads the adapted backbone once an
unfreeze has run.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "timm/vit_base_patch16_224.orig_in21k_ft_in1k"
MODEL_REVISION = "e0bd370de6799e8d1f47a911174ff4c3708e2323"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "vit-base-p16-224"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

NUM_CLASSES = 1000
INPUT_SIZE = 224  # pixels; fixed_input_size per the checkpoint config; no dynamic-resolution path
MAX_IMAGE_SIDE = 4096  # pixels; larger images are rejected before any decode-to-tensor work
MAX_BATCH = 64  # images per predict() call
DEFAULT_TOP_K = 5
DECISION_RULE = "argmax"  # the label reported as `predicted_index` is the softmax argmax; no threshold
WEIGHT_SHA256 = (
    "669b949ea91fd19217f200cee259780bde32210c1eb9a5af3859f0dd8346b2ec"  # manifest digest of WEIGHTS_FILE
)
PARAMETER_COUNT = 86_567_656
FEATURE_DIM = 768  # pre-logits width (the class token after the final norm)
TRANSFORMER_BLOCKS = 12  # ViT-B/16 depth
DEFAULT_TRAINABLE_BLOCKS = (
    2  # the unfrozen policy trains the last two blocks (14,175,744 parameters) after the probe
)
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_RECORDS = 50  # below this a scored dataset is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.vit-base-p16-224.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
POLICY_FROZEN = "frozen backbone + linear probe"
POLICY_UNFROZEN = "unfrozen last {k} blocks + linear head"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _hub_reference(model_id: str, revision: str) -> str:
    """timm's ``hf-hub:owner/name@revision`` form; ``hf_split`` passes ``revision=`` to hf_hub_download."""
    return f"hf-hub:{model_id}@{revision}"


def top_k_accuracy(predictions: Sequence[Any], targets: Sequence[int], k: int = 1) -> float:
    """Fraction of items whose target index is among the first ``k`` predicted indices.

    ``predictions`` may be the per-image dicts returned by ``predict`` or plain index sequences.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        raise ValueError("predictions must not be empty")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    hits = 0
    for pred, target in zip(predictions, targets, strict=True):
        ranked = pred["top_k"] if isinstance(pred, Mapping) else pred
        indices = [int(item["index"]) if isinstance(item, Mapping) else int(item) for item in ranked]
        hits += int(target in indices[:k])
    return hits / len(predictions)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "PIL.Image.Image or a sequence of them; any mode, converted to RGB",
    "image_side_px": [1, MAX_IMAGE_SIDE],
    "batch": [1, MAX_BATCH],
    "top_k": [1, NUM_CLASSES],
    "preprocessing": (
        "resize shorter side to 248 px, center-crop 224x224 (crop_pct 0.9, bicubic; fixed input size), "
        "normalise with mean 0.5 / std 0.5 per channel"
    ),
}


def _check_inputs(images: Any, top_k: int) -> list[Image.Image]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the images as a list."""
    if isinstance(images, Image.Image):
        images = [images]
    if not isinstance(images, Sequence) or isinstance(images, str | bytes):
        raise TypeError("images must be a PIL.Image.Image or a sequence of them")
    if not 1 <= len(images) <= MAX_BATCH:
        raise ValueError(f"batch size must be between 1 and MAX_BATCH={MAX_BATCH}, got {len(images)}")
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"each image must be a PIL.Image.Image, got {type(image).__name__}")
        width, height = image.size
        if width < 1 or height < 1 or max(width, height) > MAX_IMAGE_SIDE:
            raise ValueError(f"image side outside 1..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: {image.size}")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if not 1 <= top_k <= NUM_CLASSES:
        raise ValueError(f"top_k must be between 1 and NUM_CLASSES={NUM_CLASSES}")
    return list(images)


def validate_inputs(
    images: Image.Image | Sequence[Image.Image],
    top_k: int = DEFAULT_TOP_K,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``predict`` would; a caller that wants the
    finding recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    checked = _check_inputs(images, top_k)
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per image")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[i] if names else f"image-{i}", "mode": image.mode, "size": list(image.size)}
            for i, image in enumerate(checked)
        ],
        "top_k": top_k,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], targets: Sequence[int] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``targets`` (one ImageNet-1k index per prediction) the report carries ``top_k_accuracy``
    at k=1 and k=5 as sample-sanity evidence; without them the verdict is ``not-measurable`` and
    the report says what labelled data would make the task measurable.
    """
    predictions = result["predictions"]
    base = {
        "task": "imagenet-1k single-label classification",
        "decision_rule": result.get("decision_rule", DECISION_RULE),
        "sample_kind": sample_kind,
        "n_predictions": len(predictions),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if targets is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth class index was supplied for the evaluated images",
            "needs": (
                "labelled photographs with ImageNet-1k class indices (0-999), e.g. a held-out sample of your "
                "own data, scored with top_k_accuracy against the majority-class baseline of that sample"
            ),
        }
    top_k = int(result.get("top_k", DEFAULT_TOP_K))
    ks = sorted({1, min(5, top_k)})
    return {
        **base,
        "metrics": [
            {
                "id": "top_k_accuracy",
                "k": k,
                "value": top_k_accuracy(predictions, list(targets), k=k),
                "estimation": "single sample, no dispersion estimate",
            }
            for k in ks
        ],
        "verdict": "sample-sanity",
        "reason": f"{len(predictions)} labelled image(s) from the tutorial sample; not a benchmark",
        "needs": "a labelled evaluation set from the deployment domain for any generalisable accuracy claim",
    }


@dataclass
class ViTClassificationPipeline:
    """``_runner`` maps a float tensor (N, 3, H, W) to logits (N, NUM_CLASSES); injectable for tests."""

    _runner: Callable[[Any], Any]
    _transform: Callable[[Image.Image], Any]
    device: str = "cpu"
    labels: tuple[str, ...] = ()
    source: str = "injected"
    _feature_runner: Callable[[Any], Any] | None = field(default=None, repr=False)
    classes: list[str] | None = field(default=None, repr=False)
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _head: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ViTClassificationPipeline:
        import timm
        import torch
        from timm.data import ImageNetInfo, create_transform, resolve_model_data_config

        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        arch_name = MODEL_ID.split("/", 1)[1]
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            with open(root / CONFIG_FILE, encoding="utf-8") as fh:
                config = json.load(fh)
            snapshot_name = f"{config['architecture']}.{config['pretrained_cfg']['tag']}"
            if snapshot_name != arch_name:
                raise ValueError(f"snapshot config names {snapshot_name!r}, expected {arch_name!r}")
            overlay = dict(config["pretrained_cfg"])
            overlay["file"] = str(root / WEIGHTS_FILE)  # 'file' takes precedence over hf_hub_id in timm
            model = timm.create_model(
                arch_name, pretrained=True, pretrained_cfg_overlay=overlay, num_classes=NUM_CLASSES
            )
            source = "local-snapshot"
        elif allow_download:
            model = timm.create_model(
                _hub_reference(MODEL_ID, revision=MODEL_REVISION), pretrained=True, num_classes=NUM_CLASSES
            )
            source = "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.eval().to(resolved_device)
        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)
        info = ImageNetInfo(subset="imagenet-1k")
        labels = tuple(info.index_to_description(i) for i in range(info.num_classes()))

        def runner(batch: Any) -> Any:
            with torch.inference_mode():
                return model(batch.to(resolved_device))

        def feature_runner(batch: Any) -> Any:
            """Pre-logits: the class token after the final norm, before the ImageNet head (N, FEATURE_DIM)."""
            with torch.inference_mode():
                return model.forward_head(model.forward_features(batch.to(resolved_device)), pre_logits=True)

        return cls(
            runner, transform, resolved_device, labels, source, _feature_runner=feature_runner, _model=model
        )

    def _validate(self, images: Any, top_k: int) -> list[Image.Image]:
        return _check_inputs(images, top_k)

    def predict(
        self, images: Image.Image | Sequence[Image.Image], top_k: int = DEFAULT_TOP_K
    ) -> dict[str, Any]:
        """Classify images; ``score`` is a softmax score over 1000 classes, not a calibrated probability."""
        import torch

        batch_images = self._validate(images, top_k)
        batch = torch.stack([self._transform(image.convert("RGB")) for image in batch_images])
        logits = self._runner(batch)
        if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch_images), NUM_CLASSES):
            raise RuntimeError("runner must return a tensor of shape (batch, NUM_CLASSES)")
        scores = torch.softmax(logits.float(), dim=-1).cpu()
        values, indices = torch.topk(scores, k=top_k, dim=-1)
        predictions = []
        for image_values, image_indices in zip(values.tolist(), indices.tolist(), strict=True):
            image_values = [float(s) for s in image_values]
            ranked = [
                {"label": self.labels[i] if i < len(self.labels) else str(i), "index": i, "score": s}
                for s, i in zip(image_values, image_indices, strict=True)
            ]
            best = ranked[0]
            predictions.append(
                {"predicted_index": best["index"], "predicted_label": best["label"], "top_k": ranked}
            )
        return {
            "predictions": predictions,
            "top_k": top_k,
            "decision_rule": DECISION_RULE,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # ---- adaptation -----------------------------------------------------------------------------------

    def _require_model(self) -> Any:
        if self._model is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model

    def _require_head(self) -> tuple[Any, list[str]]:
        if self._head is None or not self.classes:
            raise ValueError("no classification head: call adapt() or load an artifact first")
        return self._head, list(self.classes)

    def _pre_logits(self, images: Sequence[Image.Image]) -> Any:
        """L2-normalised pre-logits (N, FEATURE_DIM) of validated images through the feature runner."""
        import torch

        if self._feature_runner is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        batch = torch.stack([self._transform(image.convert("RGB")) for image in images])
        features = self._feature_runner(batch)
        if not isinstance(features, torch.Tensor) or features.shape != (len(images), FEATURE_DIM):
            raise RuntimeError("feature runner must return a tensor of shape (batch, FEATURE_DIM)")
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu()

    def features(self, records: Sequence[Mapping[str, Any]]) -> Any:
        """Frozen-policy features: one L2-normalised FEATURE_DIM pre-logits vector per validated record."""
        import numpy as np

        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        rows = []
        for start in range(0, len(checked), MAX_BATCH):
            rows.append(self._pre_logits([r["image"] for r in checked[start : start + MAX_BATCH]]).numpy())
        return np.concatenate(rows, axis=0).astype(np.float32)

    def knn_baseline(
        self, train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]], *, k: int = 5
    ) -> dict[str, Any]:
        """k-NN over the frozen features of `train`, scored on `test` — the no-training reference point."""
        from .metrics import knn_baseline
        from .samples import class_names, validate_dataset

        train_checked = validate_dataset(train)["records"]
        test_checked = validate_dataset(test, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        classes = class_names(train_checked)
        return knn_baseline(
            self.features(train_checked),
            [r["label"] for r in train_checked],
            self.features(test_checked),
            [r["label"] for r in test_checked],
            classes,
            k=k,
        )

    def classify(self, images: Image.Image | Sequence[Image.Image]) -> dict[str, Any]:
        """Label validated images with the trained head over the (possibly adapted) backbone's pre-logits —
        the adaptation-contract counterpart of `predict`, which keeps the ImageNet-1k head."""
        import torch

        head, classes = self._require_head()
        batch_images = self._validate(images, DEFAULT_TOP_K)
        with torch.inference_mode():
            logits = head(self._pre_logits(batch_images))
            probabilities = torch.softmax(logits, dim=-1)
        return {
            "labels": [classes[int(i)] for i in probabilities.argmax(dim=-1)],
            "probabilities": [[float(v) for v in row] for row in probabilities.tolist()],
            "classes": classes,
            "policy": self.adapter["policy"] if self.adapter else "unknown",
            "decision_rule": DECISION_RULE,
            "device": self.device,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Score a validated labelled dataset with the trained head: accuracy, macro-F1, per-class scores."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        _head, classes = self._require_head()
        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        predictions: list[str] = []
        nll = 0.0
        index = {c: i for i, c in enumerate(classes)}
        for start in range(0, len(checked), MAX_BATCH):
            chunk = checked[start : start + MAX_BATCH]
            out = self.classify([r["image"] for r in chunk])
            predictions.extend(out["labels"])
            for record, row in zip(chunk, out["probabilities"], strict=True):
                if record["label"] not in index:
                    raise ValueError(f"label {record['label']!r} is not one of the head's classes")
                nll -= math.log(max(row[index[record["label"]]], 1e-12))
        metrics = classification_metrics([r["label"] for r in checked], predictions, classes)
        metrics.update(
            {
                "log_loss": nll / len(checked),
                "policy": self.adapter["policy"] if self.adapter else "unknown",
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _trainable_names(self, trainable_blocks: int) -> list[str]:
        if not isinstance(trainable_blocks, int) or not 0 <= trainable_blocks <= TRANSFORMER_BLOCKS:
            raise ValueError(f"trainable_blocks must be an int in 0..{TRANSFORMER_BLOCKS}")
        if trainable_blocks == 0:
            return []
        model = self._require_model()
        first = TRANSFORMER_BLOCKS - trainable_blocks
        prefixes = tuple(f"blocks.{k}." for k in range(first, TRANSFORMER_BLOCKS))
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        probe_steps: int = 300,
        probe_lr: float = 1e-2,
        trainable_blocks: int = DEFAULT_TRAINABLE_BLOCKS,
        epochs: int = 4,
        lr: float = 3e-5,
        batch_size: int = 8,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised adaptation of a validated labelled-image dataset in two stages.

        **Frozen policy** (always): a linear head over the frozen, L2-normalised class-token features is
        trained full-batch with AdamW for `probe_steps` steps (`probe_lr`, weight decay 1e-4) — the linear
        probe; epoch 0 of the history records its validation accuracy. **Unfrozen policy** (when
        `trainable_blocks` > 0): the last `trainable_blocks` transformer blocks are unfrozen and trained with
        the head end to end on the images for `epochs` epochs (AdamW at `lr`, weight decay 0.01, gradient
        clipping 1.0, seeded shuffling, no augmentation; the patch embedding, the position embedding, the
        earlier blocks and the final norm stay frozen), scored on validation after every epoch. The epoch with
        the lowest validation log-loss (mean negative log-probability of the gold label) is kept — it may be
        the probe itself; accuracy and macro-F1 are reported beside it at every epoch."""
        from .samples import class_names, validate_dataset

        if not isinstance(probe_steps, int) or not 1 <= probe_steps <= 5_000:
            raise ValueError("probe_steps must be an int in 1..5000")
        if not (0.0 < probe_lr <= 1.0):
            raise ValueError("probe_lr must be in (0, 1]")
        if not isinstance(epochs, int) or not 0 <= epochs <= 20:
            raise ValueError("epochs must be an int in 0..20")
        if not (0.0 < lr <= 1e-2):
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= MAX_BATCH:
            raise ValueError(f"batch_size must be an int in 1..{MAX_BATCH}")
        names = self._trainable_names(trainable_blocks)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        classes = class_names(train_checked)
        import torch

        torch.manual_seed(seed)
        model = self._require_model()
        started = time.perf_counter()
        device = torch.device(self.device)
        index = {c: i for i, c in enumerate(classes)}
        y_train = torch.tensor([index[r["label"]] for r in train_checked], dtype=torch.long)
        # ---- stage A: linear probe on frozen features
        x_train = torch.tensor(self.features(train_checked), dtype=torch.float32)
        head = torch.nn.Linear(FEATURE_DIM, len(classes))
        probe_opt = torch.optim.AdamW(head.parameters(), lr=probe_lr, weight_decay=1e-4)
        probe_losses = []
        for _step in range(probe_steps):
            loss = torch.nn.functional.cross_entropy(head(x_train), y_train)
            probe_opt.zero_grad(set_to_none=True)
            loss.backward()
            probe_opt.step()
            probe_losses.append(float(loss.detach()))
        head.eval()
        self._head, self.classes, self.adapter = head, classes, {"policy": POLICY_FROZEN}

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            keep = ("accuracy", "macro_f1", "log_loss", "n")
            return {k: v for k, v in self.evaluate(val_checked).items() if k in keep}

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {
            "epoch": 0,
            "stage": "linear probe (frozen backbone)",
            "train_loss": probe_losses[-1],
            "val": score_val(),
        }
        history.append(entry)
        if progress:
            progress(entry)
        best_loss = entry["val"]["log_loss"] if entry["val"] else math.inf
        wanted = set(names)
        best_state = {
            "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
            "blocks": {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted},
        }
        best_epoch = 0
        # ---- stage B: bounded unfreeze of the last blocks, end to end with the head
        n_trainable_blocks = 0
        if names and epochs > 0:
            for name, param in model.named_parameters():
                param.requires_grad_(name in wanted)
            block_params = [p for p in model.parameters() if p.requires_grad]
            n_trainable_blocks = sum(p.numel() for p in block_params)
            head.train()
            for p in head.parameters():
                p.requires_grad_(True)
            optimiser = torch.optim.AdamW(
                [{"params": block_params, "lr": lr}, {"params": list(head.parameters()), "lr": lr}],
                weight_decay=0.01,
            )
            generator = torch.Generator().manual_seed(seed)
            tensors = [self._transform(r["image"]) for r in train_checked]
            head.to(device)  # no per-batch GPU<->CPU gradient copies; moved back before the head is used
            y_device = y_train.to(device)
            initial_blocks = {k: v.clone() for k, v in best_state["blocks"].items()}
            try:
                for epoch in range(1, epochs + 1):
                    model.train()
                    head.train()
                    order = torch.randperm(len(train_checked), generator=generator).tolist()
                    losses = []
                    for start in range(0, len(order), batch_size):
                        chosen = order[start : start + batch_size]
                        batch = torch.stack([tensors[i] for i in chosen]).to(device)
                        feats = torch.nn.functional.normalize(
                            model.forward_head(model.forward_features(batch), pre_logits=True).float(), dim=-1
                        )
                        loss = torch.nn.functional.cross_entropy(head(feats), y_device[chosen])
                        optimiser.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(block_params + list(head.parameters()), 1.0)
                        optimiser.step()
                        losses.append(float(loss.detach()))
                    model.eval()
                    head.eval()
                    self.adapter = {"policy": POLICY_UNFROZEN.format(k=trainable_blocks)}
                    entry = {
                        "epoch": epoch,
                        "stage": f"unfrozen last {trainable_blocks} blocks",
                        "train_loss": sum(losses) / max(len(losses), 1),
                        "val": score_val(),
                    }
                    history.append(entry)
                    if progress:
                        progress(entry)
                    current = entry["val"]["log_loss"] if entry["val"] else -math.inf
                    if current < best_loss or not entry["val"]:
                        best_loss = current
                        best_state = {
                            "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                            "blocks": {
                                k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted
                            },
                        }
                        best_epoch = epoch
            except BaseException:
                # Transactional: a failure in training, validation or the progress callback leaves the base
                # exactly as it was, frozen, with no head or adapter attached.
                restore = dict(model.state_dict())
                restore.update(initial_blocks)
                model.load_state_dict(restore, strict=True)
                model.eval()
                for param in model.parameters():
                    param.requires_grad_(False)
                self._head, self.classes, self.adapter = None, [], None
                raise
            head.cpu()
            merged = dict(model.state_dict())
            merged.update(best_state["blocks"])
            model.load_state_dict(merged, strict=True)
            head.load_state_dict(best_state["head"])
            model.eval()
            head.eval()
            for param in model.parameters():
                param.requires_grad_(False)
        for p in head.parameters():
            p.requires_grad_(False)
        policy = POLICY_FROZEN if best_epoch == 0 or not names else POLICY_UNFROZEN.format(k=trainable_blocks)
        self._head, self.classes = head, classes
        self.adapter = {
            "policy": policy,
            "classes": classes,
            "probe_steps": probe_steps,
            "probe_lr": probe_lr,
            "probe_final_loss": probe_losses[-1],
            "trainable_blocks": trainable_blocks if names else 0,
            "trainable_names": names if best_epoch > 0 else [],
            "n_trainable_head": sum(p.numel() for p in head.parameters()),
            "n_trainable_blocks": n_trainable_blocks,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs if names else 0,
            "best_epoch": best_epoch,
            "selection": "lowest validation log-loss (epoch 0 = linear probe)"
            if val_checked
            else "final epoch",
            "lr": lr,
            "batch_size": batch_size,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the head (and any trained block tensors) as safetensors with a manifest naming the base."""
        if self.adapter is None or self._head is None:
            raise ValueError("nothing to save: call adapt() first")
        model = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter.get("trainable_names", []))
        tensors = {f"head.{k}": v.detach().cpu().contiguous() for k, v in self._head.state_dict().items()}
        tensors.update(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        )
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHTS_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter.get("history", []),
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def _check_artifact_manifest(
        self, root: Path, manifest: Mapping[str, Any]
    ) -> tuple[Path, list[str], int]:
        """Refuse an artifact whose manifest is not exactly the one this pipeline writes: the supported format
        and version, the pinned base (id, revision, weight file, digest), exactly one file entry named
        `adapter.safetensors` that resolves inside the artifact directory, at least two unique classes, a
        canonical policy and an integer `trainable_blocks` in range. Nothing is deserialised here. The
        digest check that follows detects corruption or drift of the weights relative to the adjacent
        manifest; it is not authenticity against an actor who can replace both files."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not the supported "
                f"{ARTIFACT_FORMAT_VERSION!r}"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file", WEIGHTS_FILE) != WEIGHTS_FILE:
            raise ValueError("artifact was adapted from a different base weight file")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        adapter = manifest.get("adapter")
        if not isinstance(adapter, Mapping):
            raise ValueError("artifact manifest has no adapter block")
        classes = list(adapter.get("classes") or [])
        if (
            len(classes) < 2
            or len(set(classes)) != len(classes)
            or not all(isinstance(c, str) for c in classes)
        ):
            raise ValueError("artifact manifest does not name at least two unique classes")
        blocks = adapter.get("trainable_blocks")
        if isinstance(blocks, bool) or not isinstance(blocks, int) or not 0 <= blocks <= TRANSFORMER_BLOCKS:
            raise ValueError(
                f"artifact manifest does not record an integer trainable_blocks in 0..{TRANSFORMER_BLOCKS}"
            )
        policy = adapter.get("policy")
        if policy == POLICY_FROZEN:
            blocks = 0
        elif policy != POLICY_UNFROZEN.format(k=blocks) or blocks == 0:
            raise ValueError(
                f"artifact policy {policy!r} is not a canonical policy for trainable_blocks={blocks}"
            )
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path, classes, blocks

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, rebuild
        the head and overlay its block tensors (none under the frozen policy)."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path, classes, blocks = self._check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        # The exact tensor set the recorded policy implies: the head, plus the last `blocks` blocks only.
        expected = sorted(["head.bias", "head.weight", *self._trainable_names(blocks)])
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded policy and trainable_blocks")
        model = self._require_model()
        import torch
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        if (
            tuple(tensors.get("head.weight", torch.empty(0)).shape) != (len(classes), FEATURE_DIM)
            or "head.bias" not in tensors
        ):
            raise ValueError("artifact head does not match FEATURE_DIM and the manifest's classes")
        state = model.state_dict()
        block_tensors = {k: v for k, v in tensors.items() if not k.startswith("head.")}
        for key, value in block_tensors.items():
            if key not in state or not key.startswith("blocks."):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable transformer-block tensor of the base"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key}: shape {tuple(value.shape)} != {tuple(state[key].shape)}"
                )
        head = torch.nn.Linear(FEATURE_DIM, len(classes))
        head.load_state_dict({"weight": tensors["head.weight"].float(), "bias": tensors["head.bias"].float()})
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
        if block_tensors:
            merged = dict(state)
            merged.update({k: v.to(state[k].dtype) for k, v in block_tensors.items()})
            model.load_state_dict(merged, strict=True)
            model.eval()
        self._head, self.classes = head, classes
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": sorted(block_tensors),
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ViTClassificationPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline

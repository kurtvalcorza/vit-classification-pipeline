"""Replay the fresh-test draw (owner ruling R108-D1, 2026-10-17). Data only: no model is loaded.

Rule, applied per species, in this order:
1. the pool is the fleet's 360-photograph six-species table (siglip2-vision-language-pipeline,
   src/siglip2_pipeline/samples.py at the pinned revision) minus every photograph in this repository's 180;
2. keep only photographs whose observer contributes no photograph to the 180;
3. sort by ascending iNaturalist photo id;
4. walk that order, verify each served object's byte size and SHA-256 against the pool table, and screen it for
   near-duplicates (64-bit dHash, Hamming distance <= NEAR_DUP_HAMMING, plus an exact decoded-pixel digest match)
   against the 180 and against the photographs already drawn; a hit is skipped, never kept;
5. stop at PER_SPECIES photographs; if a species cannot reach PER_SPECIES, the draw fails and pins nothing.

Usage: python tools/draw_fresh_test.py <pool samples.py> <pool revision> <cache dir> <out json>
The cache holds `<photo id>.<ext>` files; a missing or mismatched file is fetched from the open-data bucket.
"""

# ruff: noqa: E501
from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vit_classification_pipeline.samples import CORPUS_BASE_URL, SAMPLE_RECORDS  # noqa: E402

PER_SPECIES = 16
NEAR_DUP_HAMMING = 10
POOL_REPO = "kurtvalcorza/siglip2-vision-language-pipeline"
POOL_PATH = "src/siglip2_pipeline/samples.py"


def pool_records(path: Path) -> list[tuple]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        target = getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "SAMPLE_RECORDS":
            return [tuple(e.value for e in t.elts) for t in node.value.elts]
    raise ValueError(f"{path}: no SAMPLE_RECORDS table")


def served_bytes(cache: Path, photo_id: int, size: int, digest: str, ext: str) -> bytes:
    local = cache / f"{photo_id}.{ext}"
    data = local.read_bytes() if local.is_file() else b""
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        request = urllib.request.Request(
            f"{CORPUS_BASE_URL}{photo_id}/medium.{ext}", headers={"User-Agent": "dimer-vit-tutorial/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (pinned https URL)
            data = response.read()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"{photo_id}: served bytes do not match the pool pin")
        cache.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
    return data


def dhash(data: bytes, n: int = 8) -> int:
    image = Image.open(io.BytesIO(data)).convert("L").resize((n + 1, n), Image.LANCZOS)
    px = list(image.getdata())
    bits = 0
    for y in range(n):
        for x in range(n):
            bits = (bits << 1) | (px[y * (n + 1) + x] > px[y * (n + 1) + x + 1])
    return bits


def pixel_digest(data: bytes) -> str:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return hashlib.sha256(image.tobytes() + repr(image.size).encode()).hexdigest()


def main(argv: list[str]) -> int:
    pool_path, pool_revision, cache, out = Path(argv[0]), argv[1], Path(argv[2]), Path(argv[3])
    pool = pool_records(pool_path)
    in_180 = {r[2] for r in SAMPLE_RECORDS}
    observers_180 = {r[4] for r in SAMPLE_RECORDS}
    observations_180 = {r[3] for r in SAMPLE_RECORDS}
    reference = []
    for r in SAMPLE_RECORDS:
        data = served_bytes(cache, r[2], r[5], r[6], r[7])
        reference.append((r[0], dhash(data), pixel_digest(data)))
    species = sorted({r[1] for r in pool})
    drawn: list[tuple] = []
    screen = {"hamming_threshold": NEAR_DUP_HAMMING, "dhash_bits": 64, "skipped": [], "closest_kept": {}}
    counts: dict[str, dict[str, int]] = {}
    for label in species:
        candidates = sorted(
            (r for r in pool if r[1] == label and r[2] not in in_180 and r[4] not in observers_180),
            key=lambda r: r[2],
        )
        counts[label] = {
            "pool_not_in_180": sum(1 for r in pool if r[1] == label and r[2] not in in_180),
            "new_observer_candidates": len(candidates),
        }
        kept = 0
        for r in candidates:
            if kept == PER_SPECIES:
                break
            if r[3] in observations_180:
                screen["skipped"].append({"photo_id": r[2], "reason": "observation in the 180"})
                continue
            data = served_bytes(cache, r[2], r[5], r[6], r[7])
            h, p = dhash(data), pixel_digest(data)
            against = reference + [(d[0], d[-2], d[-1]) for d in drawn]
            nearest = min(against, key=lambda ref: bin(ref[1] ^ h).count("1"))
            distance = bin(nearest[1] ^ h).count("1")
            if distance <= NEAR_DUP_HAMMING or any(ref[2] == p for ref in against):
                screen["skipped"].append({"photo_id": r[2], "reason": f"near-duplicate of {nearest[0]} (dHash {distance})"})
                continue
            rid = f"fresh-{label}-{kept:02d}"
            screen["closest_kept"][rid] = {"nearest": nearest[0], "hamming": distance}
            drawn.append((rid, label, r[2], r[3], r[4], r[5], r[6], r[7], h, p))
            kept += 1
        counts[label]["drawn"] = kept
        if kept < PER_SPECIES:
            print(json.dumps({"verdict": "FAIL", "species": label, "drawn": kept, "counts": counts}))
            return 1
    records = [d[:8] for d in drawn]
    selection_digest = hashlib.sha256(
        "\n".join(f"{r[2]}:{r[5]}:{r[6]}" for r in records).encode("utf-8")
    ).hexdigest()
    payload = {
        "rule": "R108-D1 (2026-10-17): 16 per species from the fleet 360-photograph table minus the 180, observers "
        "absent from the 180 only, ascending photo id, digest-verified, near-duplicate screened",
        "pool": {"repo": POOL_REPO, "path": POOL_PATH, "revision": pool_revision, "n_records": len(pool)},
        "per_species": PER_SPECIES,
        "counts": counts,
        "near_duplicate_screen": screen,
        "n_records": len(records),
        "bytes": sum(r[5] for r in records),
        "distinct_observers": len({r[4] for r in records}),
        "selection_digest": selection_digest,
        "selection_digest_rule": "sha256 over '<photo id>:<bytes>:<sha256>' lines in record order, joined by newlines",
        "records": [dict(zip(("id", "label", "photo_id", "observation_id", "observer", "bytes", "sha256", "ext"), r, strict=True))
                    for r in records],
    }
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: payload[k] for k in ("counts", "n_records", "bytes", "distinct_observers", "selection_digest")}))
    print(json.dumps({"skipped": screen["skipped"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# Weight provenance and DIMER hosting

- Upstream: `timm/vit_base_patch16_224.orig_in21k_ft_in1k`
- Immutable revision: `e0bd370de6799e8d1f47a911174ff4c3708e2323`
- Weight format: SafeTensors (`model.safetensors`, 346284714 bytes)
- Staged files: `README.md`, `config.json`, `model.safetensors` only (the three manifest entries); no `.bin`/pickle-format file is staged or listed, so no pickle is ever deserialised.
- Upstream weight license: Apache-2.0
- Local snapshot: `weights/vit-base-p16-224/` with `dimer-base-manifest.json` (per-file bytes + SHA-256, `totalBytes` 346288708); the Git repository does not vendor the checkpoint.
- Load-time check: `verify_snapshot()` in `src/vit_classification_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to the license and notice requirements; DIMER may mirror the pinned checkpoint in its model store under the upstream license.
- Loader trust boundary: `timm==1.0.29` built-in `vit_base_patch16_224` architecture; weights loaded from a file path via `pretrained_cfg_overlay`; no remote code is executed. Hub download is opt-in and pinned to the revision above.

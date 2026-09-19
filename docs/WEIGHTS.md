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

## Adaptation corpus (tutorial data, not weights)

- Corpus: 180 research-grade iNaturalist photographs of six North American birds (Song Sparrow, Chipping Sparrow, White-throated Sparrow, Dark-eyed Junco, House Finch, American Goldfinch; 30 each, one per observer per species) whose own iNaturalist `license_code` is `cc0`, selected a priori on 2026-09-19 by the fetcher recorded in the build ledger (research grade, ordered by votes, first CC0 photo per observation) and shared with the `dinov2-feature-extraction-pipeline` tutorial. ImageNet-1k names three of the six (goldfinch, house finch, junco).
- Pins: `samples.SAMPLE_RECORDS` holds, per photograph, the record id, species key, iNaturalist photo id, observation id, observer login, byte size and SHA-256 of the served `https://inaturalist-open-data.s3.amazonaws.com/photos/<photo id>/medium.<ext>` (each photo's own served extension, `jpg` or `jpeg`, recorded in the table — 87 of the 180 are `jpeg`) (19,183,071 bytes in total). `samples.fetch_corpus` downloads each file at run time into the git-ignored `weights/inat-birds/` cache and refuses any byte or digest mismatch; `read_corpus` decodes them without re-encoding. The repository redistributes none of the photographs; every record links to its public observation page and credits its observer.
- Sample: `build_sample_dataset(seed=42)` draws 18 / 4 / 8 photographs per species (108 / 24 / 48) by a seeded stratified shuffle; `check_split_disjoint` asserts no photograph (by decoded-pixel digest) is in two splits and `observer_overlap` reports observers present in more than one split (31 of 117 in the recorded run). DIMER hosting of the weights is unaffected.
- Adapter artifacts written by the tutorial (`outputs/vit_classification_adapter/`, `org.valcorza.vit-base-p16-224.adapter.v1`, 18.6 KB when the probe is selected, about 56.7 MB with two trained blocks) carry the new linear head and, when the unfrozen policy was selected, the trained block tensors, with a manifest naming the base `model.safetensors` digest, the classes and the policy; the ImageNet-1k head is never part of them; they are outputs, not hosted weights.

# SFT_V1 10K

Portable 10,000-row multimodal SFT snapshot used for the released Qwen3.5-9B and Qwen3.5-27B checkpoints.

## Files

- `train_10000.parquet`: prompts, model-input media references, answers, source provenance and audit metadata.
- `media/`: content-addressed images referenced by paths relative to the Parquet file.
- `manifest.json`: hashes, counts, composition, source-license resolution and contamination gates.

## Composition

- 3,800 fine-grained single-image rows.
- 2,600 general visual knowledge/reasoning rows.
- 3,600 multi-image reasoning rows.

The source mixture includes Mantis-Instruct, UCSC-VLAA visual-reasoning data, aRefCOCO and VeriSciQA. Applicable licenses include Apache-2.0, CC-BY-4.0 and CC-BY-SA-4.0. Each row retains its upstream source, revision and license metadata; users are responsible for source-specific attribution and share-alike obligations.

The original machine-local Parquet SHA256 is `9f56d58c076c255df3bc660ba3c193b1cff8dd69c51ad2f73c844f5f2a8c49b0`. The portable Parquet necessarily has a different hash because all absolute paths were replaced with relative `media/` paths. See `manifest.json` for the exact portable hash.

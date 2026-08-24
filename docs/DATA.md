# SFT_V1 10K data card

SFT_V1 10K is the internally named B57 balanced fine/general/multi release used by both published SFT checkpoints. The exact original Parquet SHA256 is:

```text
9f56d58c076c255df3bc660ba3c193b1cff8dd69c51ad2f73c844f5f2a8c49b0
```

The published portable Parquet SHA256 is `bcc980bf809f6905fa9aab978e59ee8884c45258f6e69c289bf63547aa2dc859`; it references 17,459 relative content-addressed media files totaling 2,957,191,544 bytes.

## Composition

| Category | Rows | Share |
|---|---:|---:|
| Fine-grained single-image understanding | 3,800 | 38% |
| General visual knowledge/reasoning | 2,600 | 26% |
| Multi-image reasoning/QA | 3,600 | 36% |
| Total | 10,000 | 100% |

Sources include Mantis-Instruct, UCSC-VLAA visual-reasoning data, aRefCOCO and VeriSciQA. Source-specific licenses remain in force: Apache-2.0, CC-BY-4.0 and CC-BY-SA-4.0. The repository-level Apache-2.0 license applies to the release code and does not replace dataset licenses.

## Contamination controls

The publication gate excludes B28 and public Vision-OPD rows. It reports zero hard overlap against VStarBench, MMStar, BLINK and ZoomBench using exact file/RGB hashes, normalized text and exact-question plus image pHash Hamming distance at most four.

## Portable public snapshot

The original research Parquet stores machine-local absolute image paths. Publishing it unchanged would leak local paths and would not be usable elsewhere. `scripts/materialize_sft_dataset.py` therefore:

1. verifies the exact original Parquet hash and 10,000-row count;
2. hard-links or copies all model-input images into a content-addressed `media/` tree;
3. rewrites `images` and `bbox_images` to paths relative to the Parquet file;
4. removes duplicate local paths from non-training provenance fields;
5. fails if a local path or credential-like string remains;
6. writes a new portable Parquet and a manifest with its new hash.

The gold answer remains in `reward_model.ground_truth` and `extra_info.answer` because SFT consumes it. OPD does not use that answer: the released OPD command disables the reward model and distils teacher token distributions.

To create the portable snapshot:

```bash
python scripts/materialize_sft_dataset.py \
  --source-parquet /path/to/original/train_10000.parquet \
  --output /path/to/sft-v1-10k-portable \
  --mode hardlink --execute
```

The released `B54TenKSFTDataset` adapter resolves both original absolute paths and portable relative paths.

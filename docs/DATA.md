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

## Vision-OPD-6K training data

The released OPD arm uses `yuanqianhao/Vision-OPD-6K` at revision
`eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4`. The public preparation command
verifies the pinned raw files, extracts the three image roles, and writes 6,241
rows in the original order. Required training fields are:

- `images`: full image with a red target box, used by the student;
- `bbox_images`: cropped target region, used by the teacher;
- `original_images`: retained by the source/audit schema.

This is **target-conditioned perception**, not autonomous localization. The
student must read the content inside a small region from the full image, but
the red box and the prompt already reveal which region matters. The stored
`bbox` is not a prediction target, and the released OPD loss contains no bbox
or region-selection objective. This matches inference with a red-box prompt;
inference without that prompt would require additional no-box/localization
training data.

The historical full denylist audit excluded zero rows. Therefore the public
`train.parquet` and historical `train_decontaminated.parquet` select identical
training examples. Their byte SHA256 values are not a portable identity because
the Parquet rows record the resolved media root. The historical research copy
was `b8ac1cb2f17d5478af60feab2640d9526e7e816cb1506b5bd521e1598dfeb722`;
other machines should verify the pinned source revision/checksums, 6,241 rows,
row order, and view contract rather than require that path-dependent hash.

The optional `validate_vision_opd_6k.py` path performs the complete benchmark
denylist audit when the official hash-only denylist inputs are available. They
are not required to recreate the released training rows because the recorded
exclusion set is empty.

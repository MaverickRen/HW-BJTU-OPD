# Core experiment results

Frozen on 2026-08-24. Counts, totals and protocol identifiers are the primary record; percentages are derived. The machine-readable source is [`results/core_results.json`](../results/core_results.json).

## Current BLINK-v5 comparison matrix

All rows below use the same V*, MMStar and ZoomBench protocols. The BLINK column is the deterministic checkpoint-comparison v5 protocol where raw Qwen3.5-9B is `1124/1901 = 59.13%`. It is a controlled local comparison, not an official BLINK leaderboard claim.

| Model / training | V* | MMStar | BLINK-v5 | ZoomBench | Macro |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B raw | 162/191 (84.82) | 1184/1500 (78.93) | 1124/1901 (59.13) | 431/845 (51.01) | 68.47 |
| Qwen3.5-27B raw | 165/191 (86.39) | 1190/1500 (79.33) | 969/1901 (50.97) | 487/845 (57.63) | 68.58 |
| Qwen3.5-9B SFT_V1 10K | 164/191 (85.86) | 1162/1500 (77.47) | 1235/1901 (64.97) | 447/845 (52.90) | 70.30 |
| Qwen3.5-27B SFT_V1 10K | 165/191 (86.39) | 1198/1500 (79.87) | 1191/1901 (62.65) | 496/845 (58.70) | 71.90 |
| 9B SFT Student + Vision6K Crop + raw 9B teacher | 177/191 (92.67) | 1143/1500 (76.20) | 1220/1901 (64.18) | 516/845 (61.07) | 73.53 |
| 9B SFT Student + Vision6K Crop + 9B SFT teacher | 176/191 (92.15) | 1159/1500 (77.27) | **1263/1901 (66.44)** | 525/845 (62.13) | 74.50 |
| 9B SFT Student + Vision6K Crop + 27B SFT teacher | **179/191 (93.72)** | 1143/1500 (76.20) | 1255/1901 (66.02) | **537/845 (63.55)** | **74.87** |

The latest 27B-teacher run is the best macro result in this controlled matrix. Compared with raw Qwen3.5-9B, it changes V*/MMStar/BLINK-v5/ZoomBench by `+8.90/-2.73/+6.89/+12.54` points and macro by `+6.40` points. It does not improve every benchmark: MMStar remains the main regression.

## Vision-OPD checkpoint results

| Checkpoint | V* | MMStar | BLINK | ZoomBench | Status |
|---|---:|---:|---:|---:|---|
| Vision-OPD-9B official `6e41541` | 175/191 (91.62) | — | — | — | Only recovered V* is authoritative |
| Locally trained Vision-OPD-9B, B1 crop step65 | 170/191 (89.01) | 1175/1500 (78.33) | 775/1901 (40.77) | 509/845 (60.24) | Complete under BLINK-exact |

The local row uses `BLINK-exact/v14`, not `BLINK-v5`, so its BLINK number must not be compared numerically with the first table. The official checkpoint did not finish the other three formal cells; they are intentionally recorded as missing.

## Frozen protocols

- V*: `vstar_frozen_first_option_v1`, 191 questions.
- MMStar: `mmstar_qwen35_modelcard_thinking_v2`, 1,500 questions.
- BLINK-v5: `blink_deterministic_checkpoint_comparison_v5`, 1,901 questions; invalid outputs count as wrong.
- BLINK-exact: `blink_vlmevalkit_exact_matching_official_comparable_v8_blink_nonthinking`, 1,901 questions.
- ZoomBench: `zoombench_score_aggregate_v1`, 845 questions with the frozen semantic judge.

Do not merge BLINK-v5 and BLINK-exact into one ranking column. They have different prompt/generation/answer-extraction contracts.

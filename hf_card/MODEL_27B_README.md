# Qwen3.5-27B SFT_V1 10K

Merged Qwen3.5-27B checkpoint after one epoch over SFT_V1 10K: global batch 80, 125 steps, LR 1e-5, max length 9,216, bf16, seed 42, all-linear LoRA rank/alpha 16/16, then exact adapter merge.

Four-benchmark results: V* 165/191 (86.39%), MMStar 1198/1500 (79.87%), BLINK-v5 1191/1901 (62.65%), ZoomBench 496/845 (58.70%).

# Qwen3.5-9B SFT_V1 10K

Merged Qwen3.5-9B checkpoint after one epoch over SFT_V1 10K: global batch 80, 125 steps, LR 2e-5, max length 16,384, bf16, seed 42, all-linear LoRA rank/alpha 16/16, then exact adapter merge.

Four-benchmark results: V* 164/191 (85.86%), MMStar 1162/1500 (77.47%), BLINK-v5 1235/1901 (64.97%), ZoomBench 447/845 (52.90%).

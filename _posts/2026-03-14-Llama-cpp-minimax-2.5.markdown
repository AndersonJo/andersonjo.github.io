---
layout: post
title:  "Llama.cpp + Minimax 2.5 (Unsloth)"
date:   2026-03-14 01:00:00
categories: ""
asset_path: /assets/images/
tags: []
---

## Minimax M2.5

Firstly, download the following files. 

```bash
hf download unsloth/MiniMax-M2.5-GGUF \
      --include "UD-Q3_K_XL/*"
```

```bash
llama-server \
    --model ~/.cache/huggingface/hub/models--unsloth--MiniMax-M2.5-GGUF/snapshots/7c50dca0e5592483ad308ecffc876aecac725660/UD-Q3_K_XL/MiniMax-M2.5-UD-Q3_K_XL-00001-of-00004.gguf \
    --alias "unsloth/MiniMax-M2.5" \
    --prio 3 \
    --temp 0.3 \
    --top-p 0.95 \
    --min-p 0.01 \
    --top-k 40 \
    --ctx-size 32768 \
    --port 8045 \
    --flash-attn on \
    --n-gpu-layers auto \
    --threads 12
```

```bash
Vulkan0 (NVIDIA RTX PRO 6000 Blackwell Workstation Edition): 63 layers,  12412 MiB used,  84133 MiB free
Vulkan0 (NVIDIA RTX PRO 6000 Blackwell Workstation Edition): 63 layers ( 8 overflowing),  95122 MiB used,   1423 MiB free
llama_params_fit: successfully fit params to free device memory
```
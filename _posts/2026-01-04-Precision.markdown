---
layout: post
title:  "Precision and Kernel Selection - Only Hopper swizzling is supported for values"
date:   2026-01-04 01:00:00
categories: "format"
asset_path: /assets/images/
tags: ['unsloth', 'triton', 'dtype']
---



# 1. Only Hopper swizzling is supported for values

Recently I got this error, while running Unsloth with GPT-OSS-120B on Nvidia 6000 Pro blackwell. (workstation).<br>
This is due to using incorrect dtype. 

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/gpt-oss-20b-BF16',
    dtype=None,  # "bfloat16", <- it works for Nvidia RTX 6000 PRO Blackwell.
    max_seq_length=2000,
    load_in_4bit=False,
    full_finetuning=False,
    low_cpu_mem_usage=True,
    device_map="cuda"  # Explicitly load to CUDA
)
```

When I run the inference code, it raises an error, saying "only Hopper swizzling is supported."
This is because the runtime entered **Hopper-only kernel path** <br> 
but my GPU (Nvidia RTX 6000 PRO Blackwell) does not support Hopper Kernel.<br>
in short, the safest workaround is to avoid "MXFP4" and use "BF16".


# 2. Kernel Selection Flow

This is a simplified internal routing view.  
The actual checks are more granular and framework-specific.

```
Kernel Selection
├─ Op: GEMM / Linear / Matmul
│  ├─ Check device capability (SM, architecture)
│  │  ├─ Hopper? -> allow TMA/Swizzle paths
│  │  ├─ Blackwell? -> allow FP4/NVFP4 paths
│  │  └─ Other -> generic Tensor Core or CUDA paths
│  ├─ Check precision / format
│  │  ├─ FP32/TF32 -> cuBLAS / TF32-enabled GEMM
│  │  ├─ BF16/FP16 -> Tensor Core GEMM
│  │  ├─ FP8 -> FP8 kernels (often Hopper-optimized)
│  │  ├─ FP4/NVFP4 -> FP4 kernels (often Blackwell-optimized)
│  │  └─ MXFP4 -> MXFP4 kernels (specialized, high constraints)
│  ├─ Check library availability
│  │  ├─ Triton kernel exists? -> Triton path
│  │  ├─ CUTLASS/TensorRT path? -> vendor path
│  │  └─ Fallback -> cuBLAS / default GEMM
│  └─ Check runtime flags
│     ├─ load_in_4bit / quant config -> quantized kernel
│     ├─ use_cache -> cache-aware kernel
│     └─ debug/disable flags -> safe fallback
```

Practical takeaway: **selection is multi-stage**.  
If any stage assumes unsupported hardware, compilation can fail early.



# 3. Precision  

Depending on the precision (BF16/FP16/FP8/MXFP4/etc...), it selects different kernels.<br>
Here I summarized the precisions and kernels. 

```
Precision/Format
├─ FP32
│  └─ GEMM (CUDA/cuBLAS default, dtype="float32")
├─ TF32
│  └─ GEMM (Tensor Core path, dtype="float32" + TF32 enabled)
├─ FP16
│  └─ GEMM (Tensor Core path, dtype="float16")
├─ BF16
│  └─ GEMM (Tensor Core path, dtype="bfloat16", stable default)
├─ FP8 (E4M3/E5M2)
│  ├─ GEMM (FP8 kernels, dtype="float8_e4m3fn"/"float8_e5m2", Hopper-optimized)
│  └─ Swizzle/TMA (Hopper-only optimization)
├─ FP4 (NVFP4)
│  ├─ GEMM (FP4 kernels, NVFP4 format, dtype="float4"/"nvfp4", Blackwell-centered)
│  └─ Micro-tensor scaling (Blackwell-only flavor)
├─ MXFP4
│  ├─ GEMM (MXFP4 kernels, MXFP4 format)
│  └─ Swizzle/TMA (often assumes Hopper)
└─ INT8/INT4/NF4
   ├─ GEMM (int kernels, dtype="int8"/"int4", includes NF4)
   └─ Dequant (scale restore path)
```

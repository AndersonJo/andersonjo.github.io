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

# 4. Entire Error Log

```bash
# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>Solve x^5 + 3x^4 - 10 = Traceback (most recent call last):
  File "/home/anderson/projects/nlp-anderson/20_unsloth/01_gpt_oss_inference.py", line 61, in <module>
    main()
  File "/home/anderson/projects/nlp-anderson/20_unsloth/01_gpt_oss_inference.py", line 56, in main
    gptoss.generate(messages, reasoning_effort='medium')
  File "/home/anderson/projects/nlp-anderson/20_unsloth/01_gpt_oss_inference.py", line 43, in generate
    _ = self.model.generate(**inputs,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/unsloth/models/vision.py", line 300, in unsloth_base_fast_generate
    output = self._old_generate(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/transformers/generation/utils.py", line 2566, in generate
    result = decoding_method(
             ^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/transformers/generation/utils.py", line 2786, in _sample
    outputs = self(**model_inputs, return_dict=True)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/20_unsloth/unsloth_compiled_cache/unsloth_compiled_module_gpt_oss.py", line 729, in forward
    return GptOssForCausalLM_forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_router_logits, cache_position, logits_to_keep, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/_dynamo/external_utils.py", line 203, in nonrecursive_disable_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 918, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/20_unsloth/unsloth_compiled_cache/unsloth_compiled_module_gpt_oss.py", line 549, in GptOssForCausalLM_forward
    outputs: MoeModelOutputWithPast = self.model(
                                      ^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/unsloth_zoo/temporary_patches/gpt_oss.py", line 1250, in forward
    hidden_states = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/transformers/modeling_layers.py", line 94, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py", line 386, in forward
    hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
                       ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/unsloth_zoo/temporary_patches/gpt_oss.py", line 324, in mlp_forward
    routed_out = self.experts(hidden_states, routing_data, gather_idx, scatter_idx)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/unsloth_zoo/temporary_patches/gpt_oss.py", line 279, in forward
    intermediate_cache1 = matmul_ogs(
                          ^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/triton_kernels/matmul_ogs.py", line 467, in matmul_ogs
    (kernels._p_matmul_ogs if opt_flags.is_persistent else kernels._matmul_ogs)[(grid,)](
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/triton/runtime/jit.py", line 370, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/triton/runtime/jit.py", line 720, in run
    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/triton/runtime/jit.py", line 849, in _do_compile
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/triton/compiler/compiler.py", line 304, in compile
    module = src.make_ir(target, options, codegen_fns, module_map, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/anderson/projects/nlp-anderson/.venv/lib/python3.12/site-packages/triton/compiler/compiler.py", line 80, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompileTimeAssertionFailure: at 38:8:
    tl.assume(grid_n >= 0)

    is_w_microscaled: tl.constexpr = WMxScale is not None
    MX_PACK_DIVISOR: tl.constexpr = MXFP_BLOCK_SIZE
    if is_w_microscaled:
        w_type: tl.constexpr = W.dtype.element_ty
        is_mxfp4: tl.constexpr = w_type == tl.uint8
        tl.static_assert(w_type == tl.uint8 or (w_type == tl.float8e4nv or w_type == tl.float8e5),
                         "mx_weight_ptr must be uint8 or fp8")
        tl.static_assert(WMxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")
        tl.static_assert(BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR")
        tl.static_assert(SWIZZLE_MX_VALUE == "HOPPER_VALUE" or SWIZZLE_MX_VALUE is None, "Only Hopper swizzling is supported for values")
        ^
Only Hopper swizzling is supported for values
```
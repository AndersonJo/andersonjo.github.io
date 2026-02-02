---
layout: post
title: "Unsloth - Model Visualization"
date: 2026-01-02 01:00:00
categories: "unsloth"
asset_path: /assets/images/
tags: []
---


# Model Visualization

```python
from transformers import BatchEncoding, TextStreamer
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    dtype=None,  # torch.bfloat16,  # None for auto detection
    max_seq_length=1000,
    load_in_4bit=False,
    full_finetuning=False,
    low_cpu_mem_usage=True,
    device_map="cuda",  # Explicitly load to CUDA
)
```

VIsualization Codes

```python
import torch
from rich.tree import Tree
from rich import print as rprint

def visualize_model_structure(model):
    # 1. Create root node
    tree = Tree(f"🏗️ [bold blue]Model: {getattr(model.config, '_name_or_path', 'Unknown')}[/bold blue]")
    
    # Dictionary to keep track of created nodes: {path_string: rich_tree_node}
    node_lookup = {"": tree}

    for name, module in model.named_modules():
        if name == "": continue
        
        # Split path: 'model.layers.0.self_attn' -> ['model', 'layers', '0', 'self_attn']
        parts = name.split('.')
        parent_path = ".".join(parts[:-1])
        current_part = parts[-1]

        # Calculate Size Info
        # Get parameter count for this specific module
        params_count = sum(p.numel() for p in module.parameters(recurse=False))
        
        # Get shape if it's a leaf layer (like Linear or Embedding)
        shape_info = ""
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            shape_info = f" [yellow]({list(module.weight.shape)})[/yellow]"
        elif params_count > 0:
            shape_info = f" [green]({params_count:,} params)[/green]"

        # 2. Find or Create Node
        if parent_path in node_lookup:
            parent_node = node_lookup[parent_path]
            # Add new node with style and size info
            new_node = parent_node.add(f"[bold magenta]{current_part}[/bold magenta]{shape_info}")
            node_lookup[name] = new_node

    rprint(tree)

# Execution
visualize_model_structure(model)
```

here's the result

```bash
🏗️ Model: unsloth/gpt-oss-20b
├── model
│   ├── embed_tokens ([201088, 2880])
│   ├── layers
│   │   ├── 0
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 1
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 2
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 3
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 4
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 5
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 6
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 7
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 8
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 9
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 10
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 11
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 12
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 13
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 14
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 15
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 16
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 17
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 18
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 19
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 20
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 21
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   ├── 22
│   │   │   ├── self_attn (64 params)
│   │   │   │   ├── q_proj ([4096, 2880])
│   │   │   │   ├── k_proj ([512, 2880])
│   │   │   │   ├── v_proj ([512, 2880])
│   │   │   │   └── o_proj ([2880, 4096])
│   │   │   ├── mlp
│   │   │   │   ├── router ([32, 2880])
│   │   │   │   └── experts (796,538,880 params)
│   │   │   ├── input_layernorm ([2880])
│   │   │   └── post_attention_layernorm ([2880])
│   │   └── 23
│   │       ├── self_attn (64 params)
│   │       │   ├── q_proj ([4096, 2880])
│   │       │   ├── k_proj ([512, 2880])
│   │       │   ├── v_proj ([512, 2880])
│   │       │   └── o_proj ([2880, 4096])
│   │       ├── mlp
│   │       │   ├── router ([32, 2880])
│   │       │   └── experts (796,538,880 params)
│   │       ├── input_layernorm ([2880])
│   │       └── post_attention_layernorm ([2880])
│   ├── norm ([2880])
│   └── rotary_emb
└── lm_head ([201088, 2880])
```
---
layout: post
title:  "LLM - UPSTAGE SOLAR 10.7B v1.0"
date:   2024-04-13 01:00:00
categories: "llm"
asset_path: /assets/images/
tags: ['huggingface']
---


# Personal Experience

 - 한글 안되는 듯 합니다. 

# Installation

```bash
$ pip install bitsandbytes datasets accelerate peft trl
```

# Quick Code

## Import Libraries

```python
import argparse
import os
import warnings

import torch
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
```


## Loading Model and Tokenizer



```python
MODEL_ID = "Upstage/SOLAR-10.7B-v1.0"

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="./offload",
    low_cpu_mem_usage=True,
)

model.eval()
```


## Inference 


```python
text = '''
한글 가능해?
'''

inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

generation_config = GenerationConfig(
    temperature=0.1,
    max_new_tokens=256,
    
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2,
    do_sample=True,
    top_p=0.9,
    min_length=5,
    use_cache=True,
)

with torch.no_grad():
    outputs = model.generate(**inputs, generation_config=generation_config)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

일단 한글은 안됩니다. 

```text
한글 가능해?

#include <iostream>
using namespace std;
int main() {
	cout << "Hello World!" << endl; //endl은 \n과같음
	return 0;
}
```
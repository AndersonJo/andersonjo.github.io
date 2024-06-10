---
layout: post
title:  "LLM - KoAlpaca Llama 7b"
date:   2024-04-11 01:00:00
categories: "llm"
asset_path: /assets/images/
tags: ['huggingface']
---

# Personal Experience

 - python 함수를 만들어주기도 하네요. 
 - 주어진 정보를 잘 이용을 못합니다.
 - 단순한 답변에는 저장된 형태되로 뭔가 답변하긴 하는데, 퀄리티가 매우 떨어지는 느낌입니니다. 


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
MODEL_ID = "beomi/KoAlpaca-llama-1-7b"

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    offload_folder="./offload",
    llm_int8_enable_fp32_cpu_offload=True,
    low_cpu_mem_usage=True,
)
```

## Inference 


```python
# Evaluation 모드로 전환
model.eval()

prompt = """
우리는 아래와 같은 정보를 갖고 있습니다.
---------------------
한미반도체가 창사 이래 최대 규모의 단일 제품 수주 계약을 체결했다.
한미반도체는 7일 1천500억원 규모의 고대역폭메모리(HBM) 반도체 생산 장비인 ‘듀얼 TC본더 그리핀’ 납품 계약을 SK하이닉스와 체결했다고 밝혔다. 한미반도체 창사 이후 단일 제품 수주액으로는 최대 규모이며, 지난해 총 매출액(1천590억원)의 95%에 이르는 대형 계약이다.
한미반도체가 SK하이닉스와 수주 계약을 맺은 건 올해 들어 3번째다. 한미반도체는 지난 2월과 3월에도 각각 860억원과 214억원 규모의 장비를 공급하는 등 SK하이닉스와 2천500억원이 넘는 계약액을 기록했다. 한미반도체는 올해 매출 목표액을 5천500억원으로 제시했는데, 2분기 만에 3천587억원의 누적 수주액을 기록하며 순항하고 있다.
한미반도체의 주가는 이달 들어 크게 출렁였다. HBM 반도체 생산 분야 1위로 치고 나온 SK하이닉스에 독점적으로 장비를 공급해왔으나, 최근 한화정밀기계가 자체 개발한 TC 본딩 장비를 SK하이닉스에 공급할 것이란 소식이 들려오면서 한미반도체의 주가에 영향을 미쳤다. 지난 3일 한미반도체 주가는 전 거래일보다 12.99% 하락한 14만700원까지 급락하기도 했다. 한미반도체 대표인 곽동신 부회장이 다음날 자사주 30억원을 매입하는 등 적극적인 방어에 나서면서 내림세가 멈췄다.
---------------------
주어진 정보에 따라, 질문에 답해주세요.: '한미반도체의 수주를 리스트로 출력해줘'
"""

batch = tokenizer(prompt, return_tensors="pt")
prompt_size = len(batch["input_ids"][0])
print("prompt_size:", prompt_size)
batch = {k: v.to('cuda') for k, v in batch.items()}

generation_config = GenerationConfig(
    temperature=0.01,
    max_new_tokens=512,
    exponential_decay_length_penalty=(256, 1.03),
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2,
    do_sample=True,
    top_p=0.7,
    min_length=5,
    use_cache=True,
    return_dict_in_generate=True,
)

with torch.no_grad():
    generated = model.generate(**batch, generation_config=generation_config)
    response = tokenizer.decode(
        generated["sequences"][0][prompt_size:], skip_special_tokens=True
    )

print(response)
```

response 입니다.<br>
주어진 정보에서 찾지는 못하는듯 합니다. 

```text
"한미반도체의 수주: (1) SK하이닉스 (2) Samsung Electronics (3) LG Display (4) Apple Inc."
```


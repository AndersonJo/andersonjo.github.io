---
layout: post
title:  "LLM - UPSTAGE SOLAR 10.7B v1.0"
date:   2024-04-13 01:00:00
categories: "llm"
asset_path: /assets/images/
tags: ['huggingface']
---


# Personal Experience

 - 띄용? 왜 안되지? 

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
아래의 정보를 기준으로 답변하세요
-----
한미반도체가 창사 이래 최대 규모의 단일 제품 수주 계약을 체결했다.
한미반도체는 7일 1천500억원 규모의 고대역폭메모리(HBM) 반도체 생산 장비인 ‘듀얼 TC본더 그리핀’ 납품 계약을 SK하이닉스와 체결했다고 밝혔다. 한미반도체 창사 이후 단일 제품 수주액으로는 최대 규모이며, 지난해 총 매출액(1천590억원)의 95%에 이르는 대형 계약이다.
한미반도체가 SK하이닉스와 수주 계약을 맺은 건 올해 들어 3번째다. 한미반도체는 지난 2월과 3월에도 각각 860억원과 214억원 규모의 장비를 공급하는 등 SK하이닉스와 2천500억원이 넘는 계약액을 기록했다. 한미반도체는 올해 매출 목표액을 5천500억원으로 제시했는데, 2분기 만에 3천587억원의 누적 수주액을 기록하며 순항하고 있다.
한미반도체의 주가는 이달 들어 크게 출렁였다. HBM 반도체 생산 분야 1위로 치고 나온 SK하이닉스에 독점적으로 장비를 공급해왔으나, 최근 한화정밀기계가 자체 개발한 TC 본딩 장비를 SK하이닉스에 공급할 것이란 소식이 들려오면서 한미반도체의 주가에 영향을 미쳤다. 지난 3일 한미반도체 주가는 전 거래일보다 12.99% 하락한 14만700원까지 급락하기도 했다. 한미반도체 대표인 곽동신 부회장이 다음날 자사주 30억원을 매입하는 등 적극적인 방어에 나서면서 내림세가 멈췄다.
-----

한미반도체의 수주를 리스트로 뽑아줘
'''

inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

generation_config = GenerationConfig(
    temperature=0.05,
    max_new_tokens=512,
    exponential_decay_length_penalty=(256, 1.03),
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

일단 그냥 그대로 카피 페이스트하듯이 리턴합니다. <br>
사실. 왜 이렇게 리턴하는지 잘 모르겠네요. 
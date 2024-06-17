---
layout: post
title:  "LLM - MLP-TKLim Llama3 Korean Blossom 8B"
date:   2024-04-12 01:00:00
categories: "llm"
asset_path: /assets/images/
tags: ['huggingface']
---


# Personal Experience

- 뭔가 될듯 하면서, 계속 아쉬운 부분이 많이 나옵니다.

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
MODEL_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

pipeline = pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16, 
                  "offload_folder": "./offload", 
                  "low_cpu_mem_usage": True},
    device_map="auto",
)

pipeline.model.eval()
```


## Inference 


```python
PROMPT = '''
다음 문장을 자연스럽게 완성해 주세요: '인공지능은 미래의 기술로...'
'''
instruction = ""

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

print(outputs[0]['generated_text'])
```


```text
헤드라인은 다음과 같습니다:
1. 최대 규모인 단위의 반도체 생산 장비 '듀얼 TC본더 그리핀' 납품 계약을 체결했다고 합니다.
2. 한미반도체 창사 이후 단일 제품 횡령액은 최대이며, 기존 연간 순매출의 95% 이상으로 큰 계약입니다.
3. 한미반도체는 현재까지 SK하이닉스사와 2천500억원이 넘는 계약액을 기록했습니다.
4. 한미반도체는 2022년 상반기 중 3천587억원의 누진폭을 기록하다가 현재 1위 종목입니다.
5. HBM 반도체 생산 분야 1위는 현재 SK모네크로나 삼성SDI 등이 경쟁하고 있는 상황에서, SK모네크가 납품을 수행한다면 대규모 수주가 예상됩니다.
6. 현재 SK하이닉스 주식 가격은 큰 변동성을 겪고 있으며, 다음 기업의 주가가 높아질지, 하락할지에 대한 투자자들의 관심도 높아지고 있습니다.
```

## Some Examples 


민희진 대표에 대해서 설명해줘

> 민희진 대표는 국내 최초의 여성 차프터이자, 국내 자동차 산업에서 가장 성공한 여성 경영자입니다. 그녀는 수작업으로 자동차 차를 만들고, 부품의 대부분을 직접 생산하여, 국내에서 가장 신뢰받는 차 브랜드 중 하나로 자리 잡았습니다. 또한, 민희진 박사는 국내 기업들로부터 크레이지 다쉬보드의 기술을 도입하게 함과 동시에, 대규모 생산 라인을 구축함으로써 생산 효율성을 높인 것으로 알려져 있습니다.


python binary search

> 아래는 Python에서 이진 검색 알고리즘을 구현한 예시입니다.
>
> ```python
> def binarySearch(nums, target):
>     low = 0
>     high = len(nums) - 1
>     
>     while low <= high:
>         mid = (low + high) // 2
>          
>         if target == nums[mid]:
>             return mid
>         elif target < nums[mid]:
>             high = mid - 1
>         else:
>             low = mid + 1
>     
>     return -1
> ```
>
> 위의 코드에서 `binarySearch` 함수는 주어진 숫자 배열 `nums`에서 ` target` 값을 이진 검색하여 반환하는 함수입니다. 이진 검색은 반복적으로 목록을 반으로 나누고, 중간 요소를 비교하여 대상 값을 찾을 때까지 반복하는 알고리즘입니다. 이 알고리즘은 수학적 복잡도가 O(log n)으로 매우 효율적입니다.


PROMPT = '''
주어진 정보는 다음과 같습니다.
던킨도너츠 일산점에는 다음과 같은 메뉴가 있습니다. 
 - 올스타 도넛세트
 - 해피먼치킨겁 10EA
 - 버라이어티 10EA
 - 30th 해피버스데이
 - 라이스 글레이즈드
 - 인절미 후로스티드
 - (캔) 자두 쿨라타

위의 정보를 이용해서 한글 해시 태그로 사용될만한 단어를 list 형태로 출력하세요'''<br>
instruction = '#태그로 사용될만한 단어를 리스트로 여러개 추천해줘'

> - #던킨도너츠 #제주도 #이마트 #식료품 #커피 #일류 #자두 #쿨라타 #커피맛 #향상형 #모아드 #이마트오프라인
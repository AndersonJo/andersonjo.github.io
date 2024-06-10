---
layout: post
title:  "LLM - YANOLJA EEVE Korean 10.8B v1.0"
date:   2024-05-10 01:00:00
categories: "llm"
asset_path: /assets/images/
tags: ['huggingface']
---

# Personal Experience

 - 확실히 prompt 에 따라서 결과물이 좋게 나올수도 있고, 너무 엉망으로 나올때도 많습니다. 
 - 3090 x 1 에서 돌아가기 때문에 뭔가 테스트용으로 좋은 듯 합니다.


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
MODEL_ID = "yanolja/EEVE-Korean-10.8B-v1.0"

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
### 주어진 정보에 따라, 질문에 답해주세요.: '한미반도체의 수주를 리스트로 뽑아줘'
### Assistant:
"""

batch = tokenizer(prompt, return_tensors="pt")
prompt_size = len(batch["input_ids"][0])
batch = {k: v.to('cuda') for k, v in batch.items()}

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
    return_dict_in_generate=True,
)

with torch.no_grad():
    generated = model.generate(**batch, generation_config=generation_config)
    response = tokenizer.decode(
        generated["sequences"][0][prompt_size:], skip_special_tokens=True
    )

print(response)
```

response 입니다. 
```text
- [ ] 1천500억원 규모의 대규모 수주 계약
- [ ] 860억원 규모의 추가 수주 계약
- [ ] 214억원 규모의 추가 수주 계약
- [ ] 3천587억원의 누계 수주 금액 달성
```


## Cosine Similarity

일단 잘 안되요. 역시 LLM 은 generative 문제에 집중된 것이지, 이런건 그냥 Bert 로 해도 될듯요. 

```python
STORE_TEXTS = [
    "네네치킨 품동점. 가장 인기 있는 제품은 후라이드, 반반치킨이며, 그외 제품은 네네스위틱, 닭날개, 윙봉 있습니다",
    "후라이드 참 잘하는 집 풍동 식사점. 가장 인기 있는 제품은 매운양념치킨, 눈꽃 치즈치킨이며, 그외 제품은 허니버터치킨, 간장치킨 있습니다",
    "굽네치킨&피자 풍동점. 가장 인기 있는 제품은 간장치킨, 고추바사삭, 후라이드이며, 그외 제품은 웨지감자 있습니다",
    "처갓집양념치킨 백석점. 가장 인기 있는 제품은 순살 슈프림양념치킨, 반반치킨이며, 그외 제품은 매운양념치킨 있습니다",
    "이도돈까스. 가장 인기 있는 제품은 이도돈까스, 치즈돈까스이며, 그외 제품은 가께우동 있습니다",
    "돈까스왕. 가장 인기 있는 제품은 동까스왕이며, 그외 제품은 치즈까스 있습니다",
    "시골김치찌개. 가장 인기 있는 제품은 시골 김치찌개이며, 그외 제품은 도시락 있습니다",
    "전통김치찌개. 가장 인기 있는 제품은 전통김치찜, 그외 제품은 달걀말이 있습니다.",
]


def create_embeddings(model, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        # (batch, word size, d_model).
        embeddings = model(**inputs)
        return embeddings.logits.mean(dim=1).cpu().numpy()

store_embeddings = create_embeddings(model, STORE_TEXTS)
print("store_embeddings:", store_embeddings.shape)
```

```python
from sklearn.metrics.pairwise import cosine_similarity

query = '네네치킨'

query_embedding = create_embeddings(model, [query])
print("query_embedding:", query_embedding.shape)

scores = cosine_similarity(store_embeddings, query_embedding)
ranks = np.argsort(-scores.reshape(-1))

for r in ranks:
    s = scores.reshape(-1)[r]
    print(f'{s:.4f} {STORE_TEXTS[r]}')
```

네네치킨이라고 검색했는데, 대부분 값들이 다 높아요.<br>
이걸로 분별하는건 매우 어려워 보입니다. 

```text
query_embedding: (1, 32001)
0.8117 네네치킨 품동점. 가장 인기 있는 제품은 후라이드, 반반치킨이며, 그외 제품은 네네스위틱, 닭날개, 윙봉 있습니다
0.7769 처갓집양념치킨 백석점. 가장 인기 있는 제품은 순살 슈프림양념치킨, 반반치킨이며, 그외 제품은 매운양념치킨 있습니다
0.7760 굽네치킨&피자 풍동점. 가장 인기 있는 제품은 간장치킨, 고추바사삭, 후라이드이며, 그외 제품은 웨지감자 있습니다
0.7631 돈까스왕. 가장 인기 있는 제품은 동까스왕이며, 그외 제품은 치즈까스 있습니다
0.7626 이도돈까스. 가장 인기 있는 제품은 이도돈까스, 치즈돈까스이며, 그외 제품은 가께우동 있습니다
0.7595 시골김치찌개. 가장 인기 있는 제품은 시골 김치찌개이며, 그외 제품은 도시락 있습니다
0.7451 후라이드 참 잘하는 집 풍동 식사점. 가장 인기 있는 제품은 매운양념치킨, 눈꽃 치즈치킨이며, 그외 제품은 허니버터치킨, 간장치킨 있습니다
0.7428 전통김치찌개. 가장 인기 있는 제품은 전통김치찜, 그외 제품은 달걀말이 있습니다.
```
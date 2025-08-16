---
layout: post
title:  "ToolFormer - Language Models Can Teach Themselves to Use Tools"
date:   2025-08-10 01:00:00
categories: "llm"
asset_path: /assets/images/
tags: ['post-training', 'self-supervised', 'tools', 'api-calls', 'instruction-tuning']
---

# 1. Introduction

| Key              | Value                                |
|:-----------------|:-------------------------------------|
| Paper            | https://arxiv.org/pdf/2302.04761     |
| Publication Date | 2023                                  |


## 1.1 TL;DR

**Problem**

대형 언어 모델은 언어 생성 능력은 강하지만, 정확한 산술 계산, 최신 사실 조회, 날짜 연산 등에서는 취약합니다.<br> 
가장 단순한 해결책은 검색엔진·계산기·캘린더 같은 외부 도구를 쓰게 하는 것이지만, 기존 접근은 

  - (1) 대량의 사람 주석에 의존
  - (2) 특정 과제 전용 설계에 묶여 일반성이 떨어진다는 한계 존재


**Brief Solution**

ToolFormer는 최소한의 데모만으로 모델이 스스로 도구 사용을 학습하도록 합니다. 핵심 요건은 다음과 같습니다.

- LLM이 스스로 외부 API 를 호출할지, self supervised 학습.
  - 데이터 레이블을 스스로 직접 생성하고 자동으로 미세 조정
  - 실제 NLL(negative log likelihood)을 낮추는 호출을 남김
- 계산기·검색·번역·캘린더 등 다양한 도구를 하나의 통일된 포맷으로 다룸


## 1.2 Examples

 - **계산기**: [Calculator(400 / 1400) → 0.29]
 - **QA**: [QA(“Who is the publisher of The New England Journal of Medicine?”) → Massachusetts Medical Society]
 - **Wiki Search**: [WikiSearch(“BrownAct”) → The Ralph M. Brown Act is an act of the California State Legislature that guarantees the public's right to attend and participate in meetings of local legislative bodies.] 

<img src="{{ page.asset_path }}toolformer_example.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


## 1.3 Pipeline

<img src="{{ page.asset_path }}toolformer-pipeline.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">




# 2. Mathematical Formulation

## 2.1 Format & Notation

- API 호출 표현: $$ c = (a_c, i_c) $$
  - a_c: API 이름 (예: Calculator, WikiSearch)
  - i_c: API arguments (예: 질의, 수식 등)

위의 수식적 표현은 Tuple 로 이루어져 있습니다. (수학적 관점에서)<br>
하지만 Language Model에서는 문자열 즉 Token Sequence 를 다룰수 있기 때문에 Tuple 같은 구조체를 다룰수 없습니다.<br>
따라서 $$ (a_c, i_c, r) $$ 같은 구조를 -> 일렬의 문자열로 바꿔야 합니다.<br> 
이것을 논문에서는 **Linearized Sequence** 라고 부릅니다. 


$$
\begin{aligned}
\mathbf{e}(c) &= \langle \text{API} \rangle\, a_c(i_c)\, \langle /\text{API} \rangle \\
\mathbf{e}(c, r) &= \langle \text{API} \rangle\, a_c(i_c) \to r\, \langle /\text{API} \rangle
\end{aligned}
$$

여기서 <API> 는 special token입니다.



## 2.2 Sampling API Calls

1. **Prompt 를 만듬 (P(x))**
  - 각 API 마다 prompt 템플릿을 만듬
  - prompt는 모델한테 해당 텍스트 안에 API call을 생성하도록 힌트를 줌


Question Answering API Prompt 예시

```text
Your task is to add calls to a Question Answering API to a piece of text.
The questions should help you get information required to complete the text. 
You can call the API by writing "[QA(question)]" where "question" is the
question you want to ask. 

Here are some examples of API calls:

Input: Joe Biden was born in Scranton, Pennsylvania.
Output: Joe Biden was born in [QA("Where was Joe Biden born?")] Scranton, [QA("In which state is Scranton?")] Pennsylvania.
```

Calculator API Prompt 예시

```text
Your task is to add calls to a Calculator API to a piece of text.
The calls should help you get information required to complete the text. 
You can call the API by writing "[Calculator(expression)]" 
where "expression" is the expression to be computed. 

Here are some examples of API calls:

Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [Calculator(18 + 12 * 3)] 54.
```


2. **API Call 위치 후보 찾기**
 - Input Text의 i위치 마다, \<API\> 토큰이 나올 확률을 모델한테 계산
 - $$ p_i = P_M (\langle \text{API} \rangle | P(x), x_{1:i-1}) $$
 - 여기서 나온 확률이 특정 threshold 보다 크면, 해당 positions 을 들고 있음. 
 - 관련 positions 이 너무 많으면 상위 K 개만 선택함


다음은 예제 코드. (이렇게 돌리면 안됨. 그냥 예제)

```python
# ------------------------------
# 1. Prompt P(x) 준비
# ------------------------------
text = "Joe Biden was born in Scranton, Pennsylvania."
prompt = "Add helpful API calls to clarify this text:\n"  # P(x)
inputs = tokenizer(prompt + text, return_tensors="pt").to(model.device)

# ------------------------------
# 2. 각 위치별로 <API> 시작 확률 계산
# ------------------------------
input_ids = inputs["input_ids"][0]  # shape: [seq_len]
seq_len = input_ids.size(0)

# special token 정의 (논문에서는 실제론 '[' 사용)
API_START = tokenizer.convert_tokens_to_ids("[")

threshold = 0.01
top_k = 5

api_probs = []
with torch.no_grad():
    for i in range(1, seq_len):  
        prefix_ids = input_ids[:i].unsqueeze(0)  # prefix
        outputs = model(prefix_ids)
        logits = outputs.logits[:, -1, :]  # last token logits
        probs = torch.softmax(logits, dim=-1)
        p_api = probs[0, API_START].item()
        api_probs.append((i, p_api))

# threshold filtering + top-k
candidates = [(i, p) for (i, p) in api_probs if p > threshold]
candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
```


## 2.3 API 호출

여기서 전체 API 호출을 합니다.<br> 
구현에 달려 있는 것이기 때문에, python 함수를 실행할지, API 를 호출할지, 다른 model 을 호출할지 등등 모두  자유


## 2.4 Filtering API Calls


- $$ \mathcal{L}_{\text{plain}} $$ : API 호출 없이, plain LM loss
- $$ \mathcal{L}_{\text{aug}} $$ : API call + 결과를 포함한 LM Loss
- $$ \Delta^{(i)} $$ : loss 의 개선 정도  
  - $$ \Delta^{(i)} \gt 0 $$ : API Call 이 실제로 두움이 됨
  - $$ \Delta^{(i)} \ge \gamma_f $$ : API Call 이 충분히 도움이 된다고 판단 -> 해당 API 호출을 유지

 $$
\begin{aligned}
\mathcal{L}_{\text{plain}}^{(i)} &= - \sum_{t \in \mathcal{W}_i} \log p_\theta(x_t \mid x_{\le i}) \\
\mathcal{L}_{\text{aug}}^{(i)} &= - \sum_{t \in \mathcal{W}_i} \log p_\theta(x_t \mid x_{\le i}, \mathbf{e}(c_i, r)) \\
\Delta^{(i)} &= \mathcal{L}_{\text{plain}}^{(i)} - \mathcal{L}_{\text{aug}}^{(i)} \\
\end{aligned}
$$

이중에서 Delta 값이 특정 threshold 이상인 것만 남김니다. 

$$ \mathcal{L}_{\text{plain}}^{(i)} - \mathcal{L}_{\text{aug}}^{(i)} \ge \gamma_f $$







## 최종 파인튜닝 목표

필터링 후 API 호출이 삽입된 말뭉치 \(\mathcal{C}^*\) 에 대해, 표준 LM 목표를 학습합니다.

$$
\min_{\theta} \; \mathbb{E}_{\mathbf{x}^* \sim \mathcal{C}^*} \left[ - \sum_{t=1}^{|\mathbf{x}^*|} \log p_\theta\big(x_t^* \mid x_{<t}^*\big) \right]
$$

이때 \(\mathbf{x}^*\) 는 본문 토큰과 API 호출 토큰(결과 포함)이 인터리브된 시퀀스입니다. 모델은 API 호출 토큰의 문법, 어떤 API/인자를 쓸지, 결과로 이어지는 텍스트 예측까지 모두 학습하게 됩니다.


# Training Pipeline

1) In-context 후보 주석화: 소수의 데모로 프롬프트를 구성해, 말뭉치 텍스트에 API 호출 후보를 자동 삽입.
2) 실행/필터: 후보 호출을 실제 실행하여 결과를 얻고, \(\Delta^{(i)}\) 기준으로 유익한 호출만 유지.
3) 파인튜닝: 유지된 호출이 포함된 확장 말뭉치로 LM 파인튜닝.


# Core PyTorch (concise)

아래 코드는 핵심 아이디어를 담은 최소 스니펫입니다. 실제 프로덕션에서는 데이터 파이프라인, 안전한 샌드박스, 비동기 호출, batched loss 측정이 추가되어야 합니다.

```python
# Minimal, didactic snippet (PyTorch + HF Transformers)
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import math
import re

SPECIAL_OPEN = "<API>"
SPECIAL_CLOSE = "</API>"
SPECIAL_ARROW = "->"

class SafeCalculator:
    allowed = re.compile(r"^[0-9\s\+\-\*/\(\)\.]+$")
    @classmethod
    def compute(cls, expr: str) -> str:
        if not cls.allowed.match(expr):
            return "NaN"
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return "NaN"

def execute_api(name: str, arg: str) -> str:
    name = name.strip()
    if name == "Calculator":
        return SafeCalculator.compute(arg)
    if name == "Calendar":
        # Return an English string so the model learns to copy/use it
        return datetime.utcnow().strftime("Today is %A, %B %d, %Y")
    # Stubs for MT, WikiSearch etc.
    return ""

@torch.no_grad()
def sequence_nll(model, tokenizer, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, labels=enc["input_ids"])  # mean CE over tokens
    # Convert mean loss to token-summed NLL for fair window comparison
    return out.loss.item() * enc["input_ids"].numel()

def insert_api_call(text: str, i: int, api_name: str, api_arg: str, api_result: str) -> str:
    call = f"{SPECIAL_OPEN} {api_name}({api_arg}) {SPECIAL_ARROW} {api_result} {SPECIAL_CLOSE}"
    return text[:i] + call + text[i:]

def delta_loss_for_call(model, tokenizer, text: str, i: int, api_name: str, api_arg: str, window: int = 32, gamma: float = 1.0):
    # Baseline window: from i to i+window (approx by slicing raw text for simplicity)
    base_segment = text[: i + window]
    base_nll = sequence_nll(model, tokenizer, base_segment)

    result = execute_api(api_name, api_arg)
    aug_text = insert_api_call(text, i, api_name, api_arg, result)
    aug_segment = aug_text[: i + len(result) + len(api_name) + len(api_arg) + 16 + window]
    aug_nll = sequence_nll(model, tokenizer, aug_segment)

    delta = base_nll - aug_nll
    keep = (delta > gamma)
    return keep, delta, result, aug_text

class ToolformerFinetune:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.train()

    def train_step(self, batch_texts):
        enc = self.tokenizer(batch_texts, return_tensors="pt", padding=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        out = self.model(**enc, labels=enc["input_ids"])  # standard LM loss on augmented texts
        loss = out.loss
        loss.backward()
        return loss.item()

# Example (pseudo):
# corpus_text = "The number in the next term is 18 + 12 * 3 = 54."
# i = corpus_text.find("18")
# keep, delta, result, aug_text = delta_loss_for_call(
#     model=ToolformerFinetune().model,
#     tokenizer=ToolformerFinetune().tokenizer,
#     text=corpus_text,
#     i=i,
#     api_name="Calculator",
#     api_arg="18 + 12 * 3",
#     window=32,
#     gamma=0.1,
# )
# if keep:
#     ToolformerFinetune().train_step([aug_text])
```

설명
- 위 스니펫은 (a) 호출 삽입, (b) API 실행, (c) 윈도우 기반 NLL 비교에 의한 필터링, (d) API 호출이 포함된 텍스트로의 표준 LM 파인튜닝 단계를 모두 축약 구현했습니다.
- 실전에서는 문자 단위 슬라이싱 대신 토큰 경계 기반 윈도우, 배치 단위 측정, 후보 다중 샘플링과 임계치 스케줄링, API 별 포맷터/파서, 안전 샌드박싱을 추가합니다.


# Inference Behavior (at test time)

- 모델은 도구 호출 토큰을 예측할 수 있게 되며, 생성 중 필요하다고 판단되면 \(\langle \text{API} \rangle\) 토큰 시퀀스를 출력하는 경향을 학습합니다.
- 운영 환경에서는 "모델이 호출 토큰을 생성하면 실제 API를 실행하고 결과를 프롬프트에 인라인 삽입"하는 루프를 구성하면 됩니다.


# Notes

- 학습 데이터는 도구 호출이 삽입된 원-코퍼스를 사용하므로, 모델의 일반적 언어 능력을 유지하면서 도구 사용을 내재화합니다.
- 계산기/검색/번역/캘린더 등 서로 다른 API 를 동일한 프레임으로 학습할 수 있습니다.


# Reference

- Toolformer: Language Models Can Teach Themselves to Use Tools. `https://arxiv.org/pdf/2302.04761`


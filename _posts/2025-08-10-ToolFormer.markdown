---
layout: post
title:  "ToolFormer - Language Models Can Teach Themselves to Use Tools"
date:   2025-08-10 01:00:00
categories: "llm"
asset_path: /assets/images/
tags: ['post-training', 'self-supervised', 'tools', 'api-calls', 'instruction-tuning']
---

# 1. ToolFormer - Language Models Can Teach Themselves to Use Tools

| Key              | Value                                |
|:-----------------|:-------------------------------------|
| Paper            | https://arxiv.org/pdf/2302.04761     |
| Publication Date | 2023                                  |


## 1.1 Problem

대형 언어 모델은 언어 생성 능력은 강하지만, 정확한 산술 계산, 최신 사실 조회, 날짜 연산 등에서는 취약합니다. 가장 단순한 해결책은 검색엔진·계산기·캘린더 같은 외부 도구를 쓰게 하는 것이지만, 기존 접근은 

  - (1) 대량의 사람 주석에 의존
  - (2) 특정 과제 전용 설계에 묶여 일반성이 떨어진다는 한계 존재

ToolFormer는 최소한의 데모만으로 모델이 스스로 도구 사용을 학습하도록 합니다. 핵심 요건은 다음과 같습니다.

- 소수의 예시만으로 대규모 말뭉치에 도구 호출 후보를 자동 주석화하고, 실제 NLL을 낮추는 호출만 남기는 self-supervise 학습
- 모델이 스스로 언제/무엇을/어떻게 호출할지 결정하여 일반성을 유지
- API 호출과 결과를 특수 토큰으로 텍스트에 선형화해, 기존 LM 목표로 그대로 학습(언어모델링 능력 보존)
- 계산기·검색·번역·캘린더 등 다양한 도구를 하나의 통일된 포맷으로 다룸


# 1.2 Examples

 - **계산기**: [Calculator(400 / 1400) → 0.29]
 - **QA**: [QA(“Who is the publisher of The New England Journal of Medicine?”) → Massachusetts Medical Society]
 - **Wiki Search**: [WikiSearch(“BrownAct”) → The Ralph M. Brown Act is an act of the California State Legislature that guarantees the public's right to attend and participate in meetings of local legislative bodies.] 

<img src="{{ page.asset_path }}toolformer_example" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


# 1.3 Idea

- 모델이 텍스트 안에 API 호출을 삽입하고 결과까지 포함한 시퀀스로 학습하면, 모델은 다음 토큰 예측을 위해 도구 사용을 내재화합니다.
- 호출이 유익한지 여부는 "결과를 삽입했을 때 다음 토큰 NLL 감소"로 판단합니다.
- 어떤 API를 언제 호출할지는 모델이 스스로 배웁니다(특정 다운스트림 태스크로 제한되지 않음).


# Mathematical Formulation


## Format & Notation

- API 호출 표현: $$ c = (a_c, i_c) $$
  - a_c: API 이름 (예: Calculator, WikiSearch)
  - i_c: API arguments (예: 질의, 수식 등)

위의 수식적 표현은 Tuple 로 이루어져 있습니다. (수학적 관점에서)<br>
하지만 Language Model에서는 문자열 즉 Token Sequence 를 다룰수 있기 때문에 Tuple 같은 구조체를 다룰수 없습니다.
따라서 $$ (a_c, i_c, r) $$ 같은 구조를 -> 일렬의 문자열로 바꿔야 합니다.<br> 
이것을 논문에서는 **Linearized Sequence** 라고 부릅니다. 

- e(c) = <API> $$ a_c(i_c) $$ </API>
- e(c, r) = <API> $$ a_c(i_c) $$ -> r </API>

여기서 <API> 는 special token입니다.


## API 호출의 선형화 표현

API 호출 c 와 결과 r 에 대해, 다음과 같이 특수 토큰으로 선형화합니다.

$$
\begin{aligned}
\mathbf{e}(c) &= \langle \text{API} \rangle\, a_c(i_c)\, \langle /\text{API} \rangle \\
\mathbf{e}(c, r) &= \langle \text{API} \rangle\, a_c(i_c) \to r\, \langle /\text{API} \rangle
\end{aligned}
$$

- \( a_c \): API 이름 (예: Calculator, WikiSearch, MT, Calendar)
- \( i_c \): API 인자(질의, 수식 등)

## 후보 생성과 필터링

- 원시 말뭉치 \(\mathbf{x} = (x_1, \dots, x_T)\) 에서 위치 \(i\) 와 API 후보들 \(\{c_i^1, \dots, c_i^K\}\) 를 샘플링합니다(소수의 데모를 담은 프롬프트로 in-context 생성).
- 각 후보에 대해 API 를 실행해 결과 \(r\) 를 얻고, 결과를 포함한 텍스트 \(\mathbf{x}^{*}\) 를 구성합니다.
- 다음 토큰 구간 \(\mathcal{W}_i = \{i+1, \dots, i+\tau\}\) 에 대한 NLL 감소가 충분한지 검사합니다.

손실 비교 기준은 다음과 같습니다.

$$
\begin{aligned}
\mathcal{L}_{\text{plain}}^{(i)} &= - \sum_{t \in \mathcal{W}_i} \log p_\theta(x_t \mid x_{\le i}) \\
\mathcal{L}_{\text{aug}}^{(i)} &= - \sum_{t \in \mathcal{W}_i} \log p_\theta(x_t \mid x_{\le i}, \mathbf{e}(c_i, r)) \\
\Delta^{(i)} &= \mathcal{L}_{\text{plain}}^{(i)} - \mathcal{L}_{\text{aug}}^{(i)}
\end{aligned}
$$

- 필터링: \(\Delta^{(i)} > \gamma\) 인 호출만 채택합니다(임계치 \(\gamma > 0\)).

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


---
layout: post
title: "Unsloth - Lora Fine-Tuning Hyperparameters"
date: 2026-01-03 01:00:00
categories: "unsloth"
asset_path: /assets/images/
tags: [lora, fine-tuning, unsloth, llm]
---

# 1. LoRA 핵심 개념

## 1.1 문제: Full Fine-Tuning

Full Fine-Tuning은 모든 파라미터 $$ W \in \mathbb{R}^{d \times k} $$를 업데이트한다.

$$ W' = W_0 + \Delta W $$

**문제점**: 70B 모델 기준 $$\Delta W$$만 해도 140GB+ 메모리 필요.

## 1.2 LoRA의 핵심

> Fine-tuning시 weight 변화량 $$\Delta W$$는 **Low-Rank** 구조를 가진다. <br>
> **Low-Rank = 압축 가능하다**는 뜻<br>
> JPEG 압축처럼 원본 10MB → 500KB로 줄여도 품질이 비슷한 이유는, <br> 
> 정보에 **중복과 패턴**이 있기 때문.
>
> LLM Fine-tuning도 마찬가지. <br>
> 연구 결과, weight 변화량이 엄청 복잡하게 변하는 게 아니라 <br> 
> **몇 개의 주요 방향으로만 변한다**는 것을 발견 
> 이미 언어를 잘 아는 LLM에게 "의료 용어 좀 더 잘 알아듣게" 같은 미세 조정만 하면 되기 때문

즉, $$\Delta W$$를 두 개의 작은 행렬로 분해 가능:

$$\Delta W = BA$$

$$ \begin{align}
B &\in \mathbb{R}^{d \times r} \\
A &\in \mathbb{R}^{r \times k} \\
r &\ll \min(d, k)
\end{align} $$

## 1.3 파라미터 비교

| | Full Fine-Tuning | LoRA (r=8) |
|:---:|:---:|:---:|
| 파라미터 수 | $$d \times k$$ | $$d \times r + r \times k$$ |
| 예시 (4096×4096) | 16.7M | 65K |
| **압축률** | 1x | **~256x** |

---

# 2. 수학적 구조

## 2.1 Forward Pass

$$h = W_0 x + \frac{\alpha}{r} \cdot BAx$$

```
{% raw %}
Input x
    │
    ├─────────────────┐
    ▼                 ▼
┌───────┐         ┌───────┐
│  W₀   │         │   A   │ (r × k)
│frozen │         └───┬───┘
└───┬───┘             ▼
    │             ┌───────┐
    │             │   B   │ (d × r)
    │             └───┬───┘
    │                 │ × (α/r)
    ▼                 ▼
    └────────(+)──────┘
              │
              ▼
           Output h
{% endraw %}
```

### Matrix 크기 예제

구체적인 숫자로 이해해보자. `d=64, k=32, r=4`로 설정한 경우:

|        행렬         |   크기    | 파라미터 수 | 설명                    |
|:-----------------:|:-------:|:------:|:----------------------|
|      $$W_0$$      | 64 × 32 | 2,048  | 원본 weight (frozen)    |
|       $$A$$       | 4 × 32  |  128   | LoRA down-projection  |
|       $$B$$       | 64 × 4  |  256   | LoRA up-projection    |
| $$\Delta W = BA$$ | 64 × 32 |   -    | $$W_0$$와 같은 크기로 복원    |

**LoRA 파라미터**: $$A + B = 128 + 256 = 384$$ (Full의 **18.75%**)


## 2.2 초기화

- **A**: Kaiming/Gaussian 초기화
- **B**: **Zero 초기화** → 학습 시작 시 $$\Delta W = BA = 0$$

```python
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
nn.init.zeros_(self.lora_B)  # 핵심: 0으로 시작
```

## 2.3 Scaling Factor α

$$\text{scaling} = \frac{\alpha}{r}$$

| r | α | α/r | 효과 |
|:---:|:---:|:---:|:---:|
| 8 | 8 | 1.0 | 기본 |
| 8 | 16 | 2.0 | LoRA 효과 2배 |
| 16 | 16 | 1.0 | rank↑, 스케일 유지 |

---

# 3. 적용 위치 (Target Modules)

Transformer에서 LoRA 적용 가능한 위치:

```
[Attention]           [MLP (SwiGLU)]
├── q_proj  ✓         ├── gate_proj  ✓
├── k_proj  ✓         ├── up_proj    ✓
├── v_proj  ✓         └── down_proj  ✓
└── o_proj  ✓
```

**권장**: 전부 적용 (Unsloth 기본값)

---

# 4. Hyperparameters 정리

| Parameter | 권장값 | 설명 |
|:---|:---:|:---|
| `r` | 16~64 | rank. 높을수록 표현력↑, 메모리↑ |
| `lora_alpha` | r과 동일 | scaling = α/r |
| `lora_dropout` | 0 | 필요시 0.05 |
| `target_modules` | all | Attention + MLP 모두 |

---

# 5. QLoRA

Base model을 **4-bit 양자화** + LoRA:

- Base: 4-bit (frozen)
- LoRA adapters: 16-bit (trainable)

$$\text{메모리}: 140\text{GB} \rightarrow \sim24\text{GB}$$

---

# 6. Unsloth Code




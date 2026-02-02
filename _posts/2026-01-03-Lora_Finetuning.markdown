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

Full Fine-Tuning은 모든 파라미터 $$W \in \mathbb{R}^{d \times k}$$를 업데이트한다.

$$W' = W_0 + \Delta W$$

**문제점**: 70B 모델 기준 $$\Delta W$$만 해도 140GB+ 메모리 필요.

## 1.2 LoRA의 핵심 통찰

> Fine-tuning시 weight 변화량 $$\Delta W$$는 **Low-Rank** 구조를 가진다.

즉, $$\Delta W$$를 두 개의 작은 행렬로 분해 가능:

$$\Delta W = BA$$

- $$B \in \mathbb{R}^{d \times r}$$
- $$A \in \mathbb{R}^{r \times k}$$  
- $$r \ll \min(d, k)$$ (보통 8~64)

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
```

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




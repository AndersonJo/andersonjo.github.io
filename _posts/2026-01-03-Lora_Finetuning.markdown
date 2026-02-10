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

실제 Text-to-SQL 파인튜닝을 위한 코드 구현입니다.<br> 
앞서 다룬 LoRA Hyperparameters가 코드에 어떻게 적용되는지 확인합니다.

## 6.1 Model & LoRA Configuration

Scaling Factor($\frac{\alpha}{r}$)를 설정하는 단계입니다.

```python
# Initialize Model with LoRA settings
model = GptOssModel(
    model_config=ModelConfig(
        model_name="unsloth/gpt-oss-20b",
        max_seq_length=4096
    ),
    lora_config=LoRAConfig(
        r=16,           # Rank (r): SQL 로직 학습을 위해 8보다 높은 16 설정
        lora_alpha=32   # Alpha (α): Scaling factor = 32/16 = 2.0
    )
)
```
- r=16: SQL 쿼리 생성과 같은 복잡한 논리 구조를 학습하기 위해 기본값(8)보다 Rank를 높여 표현력(Expressiveness)을 확보
- lora_alpha=32: $\Delta W$의 영향력을 2배로 설정하여($\text{scaling}=2.0$), 새로운 데이터셋(SQL)의 특징을 더 강하게 반영

## 6.2 Trainer Configuration (SFT)

Unsloth의 장점인 메모리 효율성을 극대화하기 위한 SFTTrainer 설정입니다.

```python
trainer = SFTTrainer(
    model=model.model,
    processing_class=model.tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        # Memory Optimization
        per_device_train_batch_size=8,
        gradient_accumulation_steps=32,  # Effective Batch Size = 8 * 32 = 256
        
        # Optimizer & Precision
        optim="adamw_8bit",              # Optimizer State 메모리 절약
        fp16=False,
        bf16=True,                       # Ampere(RTX 30/40/6000) 이상에서 필수
        
        # Learning Rate Schedule
        learning_rate=2e-4,
        warmup_steps=5,
        max_steps=30,
        output_dir="outputs",
    ),
)
```

- gradient_accumulation_steps=32:
  - 물리적 메모리 한계로 Batch Size를 작게(8) 가져가는 대신, 32번의 step 동안 gradient를 누적해 업데이트
  - 결과적으로 대용량 배치(256)로 학습하는 것과 유사한 수렴 안정성을 확보
  - per_device_train_batch_size=8 이게 실제 batch size
  - 수식: $$\text{Effective Batch Size} = \text{Micro Batch Size} \times \text{Accumulation Steps} \times \text{Num GPUs}$$
    - $$\text{Total Batch Size} = 8 \times 32 \times 1 = \mathbf{256}$$
  - 메모리는 배치 8만큼만 쓰면서, 학습 효과는 배치 256인 것처럼 낼 수 있음
  
- optim="adamw_8bit":
  - 일반 AdamW(32-bit) 대비 Optimizer state가 차지하는 VRAM을 1/4 수준으로 줄여 OOM(Out of Memory)을 방지

- bf16=True:
  - FP16보다 표현 가능한 수의 범위(Dynamic Range)가 넓어 학습 중 발산(NaN)할 확률이 낮습니다. 
  - RTX 6000 Pro 환경에 최적화

## 6.3 Training & Saving

전체 파라미터($W$)가 아닌 LoRA Adapter($A, B$)만 학습을 진행합니다.

```python
# Start Training (Updates only A and B matrices)
trainer.train()

# Save LoRA Adapters
model.save("./text2sql_lora_model")
```
 - 학습이 끝나면 원본 모델(GB 단위)은 그대로 두고, 학습된 **LoRA weight(MB 단위)**만 저장합니다.
 - 추론 시에는 원본 모델에 이 Adapter를 동적으로 로드하여 사용하게 됩니다.
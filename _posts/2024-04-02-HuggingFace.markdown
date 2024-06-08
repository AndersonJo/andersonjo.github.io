---
layout: post
title:  "HuggingFace 101"
date:   2024-04-02 01:00:00
categories: "llm"
asset_path: /assets/images/
tags: ['huggingface']
---



# CUDA out of memory

아래와 같은 에러가 뜬다면, GPU에서 충분한 메모리를 들고 있지 않아서 입니다. 

```bash
OutOfMemoryError: CUDA out of memory. Tried to allocate 202.00 MiB. GPU 
```

## 기본적인 설정

1. python 실행전 max_split_size_mb:32 설정

```bash
# CUDA 메모리 할당시 더 작은 크기의 메모리 블록을 사용하도록 강제합니다. 
# 큰 메모리 블락시 메모리 단편화 현상이 발생가능 합니다.
#  - 메모리 단편화: 큰 블락으로 메모리 할당시 실제로는 빈공간이 남게 되며,
#    실제로는 충분한 공간이 있지만, 큰 블락의 메모리를 더이상 할당할 수 없는 상황. 
#    이를 메모리 단편화 라고 합니다. 

# Pytorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
```

2. `torch.cuda.empty_cache()` 를 모델 로딩전에 호출 

```
# 모델 로드 전에 CUDA 캐시 비우기
torch.cuda.empty_cache()
```


## 모델 로딩

| option                                | description                                                           |
|:--------------------------------------|:----------------------------------------------------------------------|
| low_cpu_mem_usage                     | 원래는 CPU -> 메모리 -> GPU인데, 바로 GPU로 업로드 / CPU 메모리 사용량 줄임                 |
| model.gradient_checkpointing_enable() | 모델이 GPU에 한번에 안 올라갈 경우 학습시 사용. 반대급부로 계산량이 많아져서 속도가 느려짐 /  올라가면 사용하지 말자 |
| llm_int8_enable_fp32_cpu_offload      | CPU 에서 FP32로 처리                                                       |



```
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder="./offload",
    llm_int8_enable_fp32_cpu_offload=True,
)

# Gradient Checkpointing 활성화
model.gradient_checkpointing_enable()

model.eval()
```





해결 방법으로 다음과 같이 합니다.<br>

1. batch_size 더 적게 조정 및  gradient_accumulation_steps=4 로 수정

```
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=4,   # batch size per device during training
    per_device_eval_batch_size=4,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    gradient_accumulation_steps=4    # Number of updates steps to accumulate before performing a backward/update pass
)
```



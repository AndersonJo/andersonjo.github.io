---
layout: post
title:  "LLM Training: Post-Training vs Fine-Tuning"
date:   2025-08-02 01:00:00
categories: "llm"
asset_path: /assets/images/
tags: ['post-training', 'fine-tuning', 'rlhf', 'sft', 'alignment', 'instruction-tuning']
---

# LLM Training: Post-Training vs Fine-Tuning

LLM (Large Language Model) 훈련은 크게 **Pre-training**, **Post-training**, 그리고 **Fine-tuning**으로 나눌 수 있습니다. 
이 글에서는 Post-training과 Fine-tuning의 차이점과 각각의 목적을 명확히 설명합니다.

# Pre-training vs Post-training vs Fine-tuning

| Stage         | Purpose                                   | Data Type               | Scale               | Techniques                   |
|:--------------|:------------------------------------------|:------------------------|:--------------------|:-----------------------------|
| Pre-training  | 언어 이해 및 생성 기초 능력 학습                       | Raw text (web, books)   | Massive (TB scale)  | Next token prediction        |
| Post-training | 인간의 선호도에 맞는 유용하고 안전한 모델로 정렬               | Human feedback, prompts | Medium (GB scale)   | SFT, RLHF, Constitutional AI |
| Fine-tuning   | 특정 task나 domain에 특화된 성능 향상                | Task-specific labeled   | Small (MB-GB)       | Supervised learning, LoRA    |

# Post-Training: 모델 정렬 (Model Alignment)

Post-training은 pre-trained 기본 모델을 인간의 가치와 선호도에 맞게 **정렬(alignment)**하는 과정입니다.

## 목적
- **유용성(Helpfulness)**: 사용자의 질문에 도움이 되는 답변 생성
- **무해성(Harmlessness)**: 유해하거나 편향된 내용 생성 방지  
- **정직성(Honesty)**: 정확하고 사실에 기반한 정보 제공

## 주요 기법들

### 1. Supervised Fine-Tuning (SFT)
고품질의 instruction-response 쌍으로 모델을 훈련

```python
# SFT 데이터 구조 예시
sft_data = [
    {
        "instruction": "파이썬에서 리스트를 정렬하는 방법을 설명해주세요.",
        "response": "파이썬에서 리스트를 정렬하는 주요 방법은 다음과 같습니다:\n1. sort() 메서드: 원본 리스트를 변경\n2. sorted() 함수: 새로운 정렬된 리스트 반환..."
    },
    {
        "instruction": "머신러닝에서 과적합을 방지하는 방법은?",
        "response": "과적합 방지 방법들:\n1. 정규화(Regularization): L1, L2 정규화\n2. 조기 종료(Early Stopping)\n3. 드롭아웃(Dropout)\n4. 데이터 증강(Data Augmentation)"
    }
]

# SFT 훈련 코드
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from datasets import Dataset

def prepare_sft_dataset(data, tokenizer, max_length=512):
    def tokenize_function(examples):
        # instruction과 response를 합쳐서 하나의 텍스트로 만듬
        texts = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            texts.append(text)
        
        # 토크나이징
        model_inputs = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # labels는 input_ids와 동일 (causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    
    # 데이터셋 준비
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# 모델과 토크나이저 로드
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# pad token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 준비
train_dataset = prepare_sft_dataset(sft_data, tokenizer)

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./sft_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no",
    learning_rate=5e-5,
    fp16=True,
)

# 트레이너 생성 및 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# SFT 훈련 실행
trainer.train()
trainer.save_model("./sft_final_model")
```

### 2. Reinforcement Learning from Human Feedback (RLHF)
인간의 선호도 피드백을 바탕으로 보상 모델을 학습하고, 이를 통해 모델을 최적화

```python
# RLHF 단계 1: 보상 모델(Reward Model) 훈련
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # 마지막 토큰의 hidden state 사용
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden_state)
        return reward

# 선호도 데이터 예시
preference_data = [
    {
        "prompt": "인공지능의 장점을 설명해주세요",
        "chosen": "AI는 반복 작업 자동화, 데이터 분석 능력 향상, 24시간 서비스 제공 등의 장점이 있습니다.",
        "rejected": "AI는 좋습니다. 끝."
    },
    {
        "prompt": "파이썬 코딩 팁을 알려주세요",
        "chosen": "1. PEP 8 스타일 가이드 준수\n2. 리스트 컴프리헨션 활용\n3. f-string 사용\n4. 적절한 변수명 사용",
        "rejected": "코딩하면 됩니다."
    }
]

def train_reward_model(model, data, tokenizer):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(3):
        for batch in data:
            # chosen과 rejected 응답을 각각 인코딩
            chosen_inputs = tokenizer(
                batch["prompt"] + " " + batch["chosen"], 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            )
            rejected_inputs = tokenizer(
                batch["prompt"] + " " + batch["rejected"], 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            )
            
            # 보상 점수 계산
            chosen_reward = model(chosen_inputs["input_ids"], chosen_inputs["attention_mask"])
            rejected_reward = model(rejected_inputs["input_ids"], rejected_inputs["attention_mask"])
            
            # Bradley-Terry 모델 손실 함수
            # P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
            loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Loss: {loss.item():.4f}")

# 보상 모델 훈련
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
reward_model = RewardModel("microsoft/DialoGPT-medium")
train_reward_model(reward_model, preference_data, tokenizer)
```

```python
# RLHF 단계 2: PPO를 사용한 정책 최적화
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, policy_model, reward_model, tokenizer):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
        
    def generate_responses(self, prompts, max_length=100):
        """정책 모델로 응답 생성"""
        responses = []
        log_probs = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            responses.append(response)
            
            # log probability 계산 (간소화된 버전)
            logits = torch.stack(outputs.scores, dim=1)
            log_prob = F.log_softmax(logits, dim=-1).mean()
            log_probs.append(log_prob)
            
        return responses, log_probs
    
    def compute_rewards(self, prompts, responses):
        """보상 모델로 보상 계산"""
        rewards = []
        for prompt, response in zip(prompts, responses):
            full_text = prompt + " " + response
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                reward = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
            rewards.append(reward.item())
            
        return rewards
    
    def ppo_update(self, log_probs, rewards, old_log_probs):
        """PPO 알고리즘으로 정책 업데이트"""
        epsilon = 0.2  # PPO clipping parameter
        
        # Advantage 계산 (간소화)
        advantages = torch.tensor(rewards) - torch.tensor(rewards).mean()
        
        for log_prob, old_log_prob, advantage in zip(log_probs, old_log_probs, advantages):
            # Importance sampling ratio
            ratio = torch.exp(log_prob - old_log_prob)
            
            # PPO clipped objective
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

# PPO 훈련 실행
def train_with_ppo(policy_model, reward_model, tokenizer, prompts, epochs=5):
    trainer = PPOTrainer(policy_model, reward_model, tokenizer)
    
    for epoch in range(epochs):
        # 1. 응답 생성
        responses, log_probs = trainer.generate_responses(prompts)
        
        # 2. 보상 계산  
        rewards = trainer.compute_rewards(prompts, responses)
        
        # 3. PPO 업데이트
        old_log_probs = [lp.detach() for lp in log_probs]
        trainer.ppo_update(log_probs, rewards, old_log_probs)
        
        print(f"Epoch {epoch}: Average Reward = {sum(rewards)/len(rewards):.3f}")

# 실행 예시
prompts = ["파이썬 코딩 팁을 알려주세요", "인공지능의 장점을 설명해주세요"]
policy_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
train_with_ppo(policy_model, reward_model, tokenizer, prompts)
```

### 3. Constitutional AI
모델이 스스로 자신의 출력을 비판하고 개선하도록 훈련

```python
# Constitutional AI 구현 예시
class ConstitutionalAI:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Constitution (헌법) - 모델이 따라야 할 원칙들
        self.constitution = [
            "응답은 도움이 되고 정확해야 합니다.",
            "유해하거나 차별적인 내용을 포함하면 안 됩니다.",
            "사실에 기반한 정보를 제공해야 합니다.",
            "예의 바르고 존중하는 톤을 유지해야 합니다."
        ]
    
    def generate_initial_response(self, prompt):
        """초기 응답 생성"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
    
    def critique_response(self, prompt, response):
        """응답 비판하기"""
        critique_prompt = f"""
다음 질문과 답변을 평가해주세요:

질문: {prompt}
답변: {response}

다음 기준에 따라 이 답변의 문제점을 지적해주세요:
1. 도움이 되고 정확한가?
2. 유해하거나 차별적인 내용이 있는가?
3. 사실에 기반한 정보인가?
4. 예의 바르고 존중하는 톤인가?

문제점 (있다면):
"""
        
        inputs = self.tokenizer(critique_prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=300,
            do_sample=True,
            temperature=0.3,
            pad_token_id=self.tokenizer.eos_token_id
        )
        critique = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return critique.replace(critique_prompt, "").strip()
    
    def revise_response(self, prompt, original_response, critique):
        """응답 수정하기"""
        revision_prompt = f"""
원본 질문: {prompt}
원본 답변: {original_response}
문제점: {critique}

위 문제점들을 고려하여 더 도움이 되고, 안전하며, 정확한 답변으로 수정해주세요:

수정된 답변:
"""
        
        inputs = self.tokenizer(revision_prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=250,
            do_sample=True,
            temperature=0.5,
            pad_token_id=self.tokenizer.eos_token_id
        )
        revised = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return revised.replace(revision_prompt, "").strip()
    
    def constitutional_process(self, prompt, max_iterations=3):
        """Constitutional AI 전체 프로세스"""
        print(f"질문: {prompt}\n")
        
        # 1. 초기 응답 생성
        response = self.generate_initial_response(prompt)
        print(f"초기 응답: {response}\n")
        
        for iteration in range(max_iterations):
            # 2. 응답 비판
            critique = self.critique_response(prompt, response)
            print(f"비판 {iteration+1}: {critique}\n")
            
            # 비판에서 문제점이 없다고 판단되면 종료
            if "문제없음" in critique or "적절함" in critique:
                print("Constitutional AI 프로세스 완료: 문제점 없음")
                break
            
            # 3. 응답 수정
            revised_response = self.revise_response(prompt, response, critique)
            print(f"수정된 응답 {iteration+1}: {revised_response}\n")
            
            response = revised_response
        
        return response

# Constitutional AI 실행 예시
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# pad token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

constitutional_ai = ConstitutionalAI(model, tokenizer)

# 테스트 실행
prompt = "인공지능의 위험성에 대해 설명해주세요"
final_response = constitutional_ai.constitutional_process(prompt)
print(f"최종 응답: {final_response}")
```

```python
# Constitutional AI를 위한 Self-Supervised Training
def train_constitutional_ai(model, tokenizer, training_prompts, epochs=3):
    """
    Constitutional AI 방식으로 모델을 훈련
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    constitutional_ai = ConstitutionalAI(model, tokenizer)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for prompt in training_prompts:
            # 1. 초기 응답 생성
            initial_response = constitutional_ai.generate_initial_response(prompt)
            
            # 2. 자기 비판 및 수정
            final_response = constitutional_ai.constitutional_process(prompt, max_iterations=2)
            
            # 3. 개선된 응답으로 모델 훈련
            # 초기 응답보다 수정된 응답을 더 높은 확률로 생성하도록 학습
            improved_text = f"{prompt} {final_response}"
            
            # 토크나이징
            inputs = tokenizer(
                improved_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(training_prompts)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    return model

# Constitutional AI 훈련 데이터 예시
training_prompts = [
    "인공지능의 위험성에 대해 설명해주세요",
    "프로그래밍을 배우는 가장 좋은 방법은?",
    "기후 변화의 주요 원인은 무엇인가요?",
    "건강한 식습관에 대한 조언을 해주세요"
]

# Constitutional AI 훈련 실행
trained_model = train_constitutional_ai(model, tokenizer, training_prompts)
```

# Fine-Tuning: 특화 성능 향상

Fine-tuning은 특정 작업이나 도메인에 특화된 성능을 향상시키는 과정입니다.

## 목적
- **Task-specific 성능**: 번역, 요약, 분류 등 특정 작업 성능 향상
- **Domain adaptation**: 의료, 법률, 금융 등 전문 분야 적응
- **Style/Format**: 특정 출력 형식이나 스타일 학습

## 주요 기법들

### 1. Full Fine-tuning
모델의 모든 파라미터를 업데이트

```python
# Full Fine-tuning 완전한 구현 예시
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

# 1. Task-specific 데이터 준비 (요약 작업 예시)
summarization_data = [
    {
        "text": "최근 인공지능 기술의 발전이 빠르게 진행되고 있다. 특히 대규모 언어모델들이 다양한 작업에서 인간 수준의 성능을 보여주고 있으며, 이는 자연어 처리 분야에 혁신을 가져오고 있다.",
        "summary": "인공지능, 특히 대규모 언어모델이 자연어 처리 분야에 혁신을 가져오고 있다."
    },
    {
        "text": "기후변화는 전 세계적인 문제로, 탄소배출 감소와 재생에너지 활용이 필요하다. 각국 정부와 기업들이 친환경 정책을 도입하고 있으며, 개인들도 일상생활에서 환경을 고려한 선택을 하고 있다.",
        "summary": "기후변화 대응을 위해 정부, 기업, 개인이 친환경 정책과 행동을 실천하고 있다."
    },
    {
        "text": "원격근무가 일반화되면서 업무 효율성과 워라밸에 대한 관심이 높아졌다. 기업들은 새로운 협업 도구를 도입하고, 직원들은 홈오피스 환경을 개선하고 있다.",
        "summary": "원격근무 확산으로 업무 효율성과 워라밸에 대한 관심이 증가했다."
    }
]

def prepare_full_finetuning_dataset(data, tokenizer, max_length=512):
    def tokenize_function(examples):
        # 입력: "요약해주세요: [텍스트]" / 출력: "[요약문]"
        inputs = [f"다음 텍스트를 요약해주세요: {text}" for text in examples['text']]
        targets = examples['summary']
        
        # 입력-출력을 하나의 텍스트로 결합
        full_texts = [f"{inp}\n\n요약: {target}" for inp, target in zip(inputs, targets)]
        
        # 토크나이징
        model_inputs = tokenizer(
            full_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # labels 설정 (causal LM이므로 input_ids와 동일)
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        # 입력 부분은 loss 계산에서 제외 (선택사항)
        # 실제 구현에서는 instruction 부분을 마스킹할 수 있음
        
        return model_inputs
    
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# 2. 모델과 토크나이저 준비
model_name = "microsoft/DialoGPT-medium"  # 실제로는 더 큰 모델 사용
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 메모리 효율성을 위해
    device_map="auto"
)

# pad token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. 데이터셋 준비
train_dataset = prepare_full_finetuning_dataset(summarization_data, tokenizer)

# 4. 훈련 설정
training_args = TrainingArguments(
    output_dir="./full_finetuned_model",
    num_train_epochs=5,
    per_device_train_batch_size=2,  # GPU 메모리에 따라 조정
    gradient_accumulation_steps=4,  # 효과적인 배치 크기 = 2 * 4 = 8
    warmup_steps=50,
    weight_decay=0.01,
    learning_rate=2e-5,  # Full fine-tuning에서는 더 작은 학습률 사용
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    dataloader_drop_last=True,
    fp16=True,  # 혼합 정밀도 훈련
    gradient_checkpointing=True,  # 메모리 절약
    report_to=None,  # wandb 등 로깅 도구 비활성화
)

# 5. Data Collator 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM이므로 MLM 사용 안 함
)

# 6. Trainer 설정 및 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Full Fine-tuning 시작...")
trainer.train()

# 7. 모델 저장
trainer.save_model("./full_finetuned_summary_model")
tokenizer.save_pretrained("./full_finetuned_summary_model")

print("Full Fine-tuning 완료!")

# 8. 훈련된 모델 테스트
def test_finetuned_model(model, tokenizer, test_text):
    input_text = f"다음 텍스트를 요약해주세요: {test_text}\n\n요약:"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + 50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = response.split("요약:")[-1].strip()
    return summary

# 테스트
test_text = "머신러닝은 데이터로부터 학습하는 인공지능의 한 분야이다. 지도학습, 비지도학습, 강화학습으로 나뉘며, 각각 다른 방식으로 모델을 훈련한다."
summary = test_finetuned_model(model, tokenizer, test_text)
print(f"생성된 요약: {summary}")
```

### 2. Parameter-Efficient Fine-tuning (PEFT)
일부 파라미터만 업데이트하여 효율성 향상

**LoRA (Low-Rank Adaptation)**
```python
# LoRA 완전한 구현 예시
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
import torch

# 1. QLoRA 설정 (4bit 양자화 + LoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. 기본 모델 로드 (4bit 양자화로)
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# pad token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. LoRA 설정
lora_config = LoraConfig(
    r=16,  # rank - 낮을수록 파라미터 적음, 높을수록 표현력 높음
    lora_alpha=32,  # scaling factor (일반적으로 r의 2배)
    target_modules=[
        "c_attn",  # DialoGPT의 attention projection layers
        "c_proj",  # DialoGPT의 output projection
        "c_fc",    # DialoGPT의 feed-forward layers
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 4. LoRA 모델 생성
model = prepare_model_for_kbit_training(model)
lora_model = get_peft_model(model, lora_config)

# 훈련 가능한 파라미터 확인
lora_model.print_trainable_parameters()
# 출력 예시: "trainable params: 1,048,576 || all params: 355,804,160 || trainable%: 0.29%"

# 5. 훈련 데이터 준비 (의료 도메인 예시)
medical_data = [
    {
        "instruction": "고혈압의 증상은 무엇인가요?",
        "response": "고혈압의 주요 증상으로는 두통, 어지러움, 목 뒤쪽 통증, 시야 흐림, 코피 등이 있습니다. 하지만 많은 경우 증상이 없어 '조용한 살인자'라고 불립니다."
    },
    {
        "instruction": "당뇨병 환자의 식이요법은?",
        "response": "당뇨병 환자는 규칙적인 식사, 탄수화물 조절, 섬유질 섭취 증가, 단순당 제한이 필요합니다. 혈당 지수가 낮은 음식을 선택하고 적절한 칼로리를 유지해야 합니다."
    },
    {
        "instruction": "심장마비의 응급처치 방법은?",
        "response": "심장마비 응급처치: 1) 즉시 119 신고 2) 환자를 평평한 곳에 눕히기 3) 기도 확보 4) 심폐소생술(CPR) 실시 5) AED 사용 가능시 사용"
    }
]

def prepare_lora_dataset(data, tokenizer, max_length=512):
    def format_instruction(example):
        return f"### 질문: {example['instruction']}\n### 답변: {example['response']}"
    
    def tokenize_function(examples):
        texts = [format_instruction(ex) for ex in examples]
        
        model_inputs = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    
    # 리스트를 Dataset으로 변환
    from datasets import Dataset
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function([x]), 
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

# 6. 데이터셋 준비
train_dataset = prepare_lora_dataset(medical_data, tokenizer)

# 7. 훈련 설정
training_args = TrainingArguments(
    output_dir="./lora_medical_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=20,
    weight_decay=0.01,
    learning_rate=2e-4,  # LoRA는 더 높은 학습률 사용 가능
    logging_dir="./logs",
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    evaluation_strategy="no",
    fp16=False,  # 4bit 양자화 사용시 fp16 비활성화
    bf16=True,
    gradient_checkpointing=True,
    dataloader_drop_last=True,
    report_to=None,
)

# 8. Trainer 설정 및 훈련
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

print("LoRA Fine-tuning 시작...")
trainer.train()

# 9. LoRA 어댑터 저장
lora_model.save_pretrained("./lora_medical_adapter")
print("LoRA 어댑터 저장 완료!")

# 10. 훈련된 LoRA 모델 테스트
def test_lora_model(model, tokenizer, question):
    input_text = f"### 질문: {question}\n### 답변:"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + 100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("### 답변:")[-1].strip()
    return answer

# 테스트
test_question = "폐렴의 증상과 치료법을 알려주세요"
answer = test_lora_model(lora_model, tokenizer, test_question)
print(f"질문: {test_question}")
print(f"답변: {answer}")

# 11. LoRA 어댑터 로드하는 방법
# from peft import PeftModel
# base_model = AutoModelForCausalLM.from_pretrained(model_name)
# lora_model = PeftModel.from_pretrained(base_model, "./lora_medical_adapter")
```

**Adapter Layers**
```python
# Adapter 구현 예시
import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    def __init__(self, input_dim, adapter_dim=64, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        
        # Down projection: input_dim -> adapter_dim
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Up projection: adapter_dim -> input_dim
        self.up_proj = nn.Linear(adapter_dim, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        # 원본 입력 저장 (잔차 연결용)
        residual = x
        
        # Adapter 연산: down -> activation -> up
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        
        # 잔차 연결: 원본 + adapter 출력
        return residual + x

class TransformerWithAdapter(nn.Module):
    """Adapter가 포함된 Transformer 예시"""
    def __init__(self, base_model, adapter_dim=64):
        super().__init__()
        self.base_model = base_model
        
        # 기존 모델의 파라미터 고정
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Adapter layers 추가
        self.adapters = nn.ModuleDict()
        
        # 각 transformer layer에 adapter 추가
        for name, module in self.base_model.named_modules():
            if "attention" in name.lower() or "mlp" in name.lower():
                # attention과 MLP 뒤에 adapter 추가
                hidden_size = getattr(module, 'hidden_size', 768)  # 기본값
                self.adapters[name] = AdapterLayer(hidden_size, adapter_dim)
    
    def forward(self, *args, **kwargs):
        # 이 부분은 실제 모델 구조에 맞게 수정 필요
        outputs = self.base_model(*args, **kwargs)
        
        # Adapter 적용 (실제 구현에서는 hook이나 다른 방법 사용)
        # 여기서는 개념적 예시만 제공
        
        return outputs

# Adapter 훈련 예시
def train_adapter_model(base_model, train_data, tokenizer):
    # Adapter 모델 생성
    adapter_model = TransformerWithAdapter(base_model, adapter_dim=64)
    
    # 훈련 가능한 파라미터 확인
    trainable_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in adapter_model.parameters())
    
    print(f"훈련 가능한 파라미터: {trainable_params:,}")
    print(f"전체 파라미터: {total_params:,}")
    print(f"훈련 비율: {100 * trainable_params / total_params:.2f}%")
    
    # 훈련 설정
    optimizer = torch.optim.AdamW(adapter_model.parameters(), lr=1e-4)
    
    # 실제 훈련 루프는 데이터와 모델 구조에 따라 구현
    # 여기서는 개념적 예시만 제공
    
    return adapter_model

# 사용 예시
# base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
# adapter_model = train_adapter_model(base_model, train_data, tokenizer)
```

# Post-Training vs Fine-Tuning 비교

## 언제 사용할까?

### Post-Training이 필요한 경우
- 모델이 유해한 내용을 생성하는 경우
- 사용자 지시를 잘 따르지 않는 경우
- 일반적인 대화 능력이 부족한 경우
- 편향이나 부정확한 정보를 자주 생성하는 경우

### Fine-tuning이 필요한 경우  
- 특정 작업 (번역, 요약, 분류)의 성능을 높이고 싶은 경우
- 특정 도메인 (의료, 법률)에 특화시키고 싶은 경우
- 특정 출력 형식이나 스타일이 필요한 경우
- 제한된 데이터로 빠른 적응이 필요한 경우

## 데이터 요구사항

### Post-Training
```python
# RLHF 선호도 데이터 예시
{
    "prompt": "인공지능의 위험성에 대해 설명해주세요",
    "chosen": "AI는 잘못 사용될 경우 개인정보 침해, 편향 증폭 등의 위험이...",
    "rejected": "AI는 위험하니까 사용하지 마세요"
}
```

### Fine-tuning  
```python
# Task-specific 데이터 예시
{
    "input": "다음 문서를 요약해주세요: [긴 텍스트]",
    "output": "[3-4줄 요약문]"
}
```

# 실제 적용 예시

## GPT-4 개발 과정
```text
1. Pre-training: 웹 텍스트로 언어 모델 학습
2. Post-training: 
   - SFT로 instruction following 학습
   - RLHF로 인간 선호도 정렬
3. Fine-tuning: 
   - Code generation (GitHub Copilot)
   - 특정 API 호출 형식 학습
```

## Domain-specific 모델 개발
```text
1. Base Model (Llama-2)
2. Post-training: 일반적인 안전성/유용성 정렬
3. Fine-tuning: 의료 텍스트 데이터로 MedLlama 개발
```

# 결론

- **Post-training**은 모델을 인간의 가치와 **정렬**하는 과정
- **Fine-tuning**은 특정 작업이나 도메인에 **특화**시키는 과정
- Post-training은 모델의 전반적인 행동을 개선하고, Fine-tuning은 특정 성능을 향상
- 실제 상용 LLM은 두 과정을 모두 거쳐 개발됨

각 단계는 서로 다른 목적과 기법을 가지며, 최종적으로 안전하고 유용하며 특화된 성능을 갖춘 LLM을 만들기 위해 모두 필요한 과정입니다.
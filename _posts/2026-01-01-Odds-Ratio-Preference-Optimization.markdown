---
layout: post
title: "Unsloth - Odds Ratio Preference Optimization (ORPO)"
date: 2026-01-01 01:00:00
categories: "unsloth"
asset_path: /assets/images/
tags: [ ]
---

# 1. What is Post Training

Pre-training을 통해서 언어 자체를 배웁니다. <br>
간단하게 설명하면, GPT계열 (Llama 3, Mistral, Gemma) 등은, 앞의 문장을 보고 다음 단어를 맞추는 방식으로 학습합니다.
보통 생성형 계열에서 많이 합니다. <br>
Masked Language Modeling 방식은 빈칸 채우기 인데 Bert, RoBERTa 계열에서 많이 쓰입니다. <br>
문장의 의미를 파악하거나, 분류하는데 좋지만, 긴 글을 지어내는 능력은 떨어집니다.

Post Training 은 이렇게 언어 자체에 대해서 학습한 모델에 대해서<br>
원하는 목적에 맞게 튜닝하는 과정입니다.

## 1.1 Supervised Fine Tuning

초기 이렇게 Post Training을 했습니다.

- Supervised Fine-Tuning (SFT)
    - Instruction (질문) 과 Response (모범 답안) 쌍으로 이루어진 데이터로 학습
    - Next token prediction 을 사용하되, 데이터 품질과 도메인 지식을 따라하도록 학습
    - Limitation: 해보면, 흉내는 냄. 근데 이게 안전한지, 유용한지에 대한 가치 판단을 하지 못함, 
    - Hallucination 심함


## 1.2 LLM Alignment 

이전에는 지식을 그냥 넣었다면, 사람이 원하는 목적에 맞게 다듬는 과정 

- [1세대] Reinforcement Learning from Human Feedback (RLHF)
  - ChatGPT 에서 만든 정석 방법
  - 사람이 매긴 점수를 바탕으로 보상 모델 (Reward Model) 을 만들고, 이를 통해 강화학습 (PPO) 수행
  - 성능은 정말 압도적. 실제 인간과 대화하는 듯한 자연스러운 성능
  - Limitation
    - 학습 과정 자체가 너무 복잡
    - GPU 메모리를 많이 사용 -> 개인이 시도하기에는 어려움
    - 학습시 모4개의 모델을 메모리에 올려야 함
      - Policy Model (학습 대상)
      - Reference Model (비교 대상)
      - Reward Model (점수 매김)
      - Critical Model (가치 판단 - PPO 내부용)
    
- [2세대] Direct Preference Optimization (DPO)
  - 선호하는 답변 (Choosen), 거부된 답변 (Rejected) 쌍을 이용해서 학습
  - 현재 업계 표준. 학습이 훨씬 안정적으로 빠름
  - Limitation
    - 여전히 SFT 가 선행되야 함 (2-stage)
    - 학습시 2개의 모델이 필요
      - Policy Model (실제 학습하는 모델) 
      - Reference Model (비교 대상)

- [3세대] Odds Ratio Preference Optimization (ORPO)
  - SFT + Alignment 를 하나로 합쳤음 (1-Stage)
  - Reference Model 이 필요없음
  - 한정된 자원으로 (GPU VRAM) 으로 학습시킬때 이 방식이 최고
  - Loss Function 안에서 직접 Rejected Answer에 대한 Penalty를 사용
  - Unsloth 추천
  - Limitation
    - SFT + Alignment를 동시에 하기 때문에 -> Chosen 답변이 정말 모범 답안이어야 함
    - 수렴이 잘 안되기도 함, hyperparameter에 따라서 학습 불안정
    - 

- [4세대] Group Relative Policy Optimization (GRPO)
  - DeepSeek-V3 그리고 R1 으로 대중화 됨
  - RLHF의 강력한 성능은 유지
  - 하나의 질문에 여러개의 답변 **그룹**을 생성 -> 그 안에서 상대적인 점수를 매김
  - 평균보다 잘한 답변은 강화, 못한 답변은 멀어지게 만듬
  - 메모리 효율
    - RLHF처럼 4개의 모델을 띄울 필요 X
    - Policy Model 1개 학습시 필요
    - 비교를 위해서 Reference Model을 vLLM등으로 따로 띄워서 사용 (Optional)
  - Limitation
    - Group Size 에 따른 VRAM 부담 -> 8개 16개 답변을 동시 생성해야됨 (많은 자원 소모)
    - 명확한 보상 함수 필요 
      - 친절한 말투 -> 이런 모호한건 학습 잘 안됨
      - 수학문제처럼 명확한거를 잘함


가장 중요한 메모리 필요 부분을 정리하면
 - RLHF: 🟥🟥🟥🟥 (모델 4개 분량) - Out of Memory!
 - DPO: 🟦🟦 (모델 2개 분량) - Heavy
 - ORPO: 🟩 (모델 1개 분량) - Lightweight & Fast 
 - GRPO: 🟨🟨 (모델 1~2개 수준 - RLHF 대비 훨씬 가벼움)

> 해당 문서에서는 ORPO 를 기술적 방법을 설명합니다. 

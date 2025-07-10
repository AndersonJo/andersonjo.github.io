---
layout: post
title: "Switch Transformer - Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
date: 2025-06-01 01:00:00
categories: "nlp"
asset_path: /assets/images/
tags: [ ]
---

# Switch Transformer - Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

| Key              | Value                            |
|:-----------------|:---------------------------------|
| Paper            | https://arxiv.org/pdf/2101.03961 |
| publication Date | 2021                             |


# Problem

MoE (Mixture of Experts) 모델은 inputs 마다 서로다른 parameters 사용하기 때문에,
전체 parameters 수는 폭발적으로 늘어나도, 실제 사용하는 부분은 소수라 연산량은 일정합니다.

하지만 실제 적용은 어렵습니다. 
- 기존 MoE 모델의 문제점
    - 복잡한 Routing 알고리즘이 필요
    - communication overhead가 큼
    - 학습 불안정성 gradient exploding 현상 발생
- Switch Transformer 는 이러한 기존의 문제들을 해결하였음. 

# Model 

핵심은
 - Sparse Training 
 - 궁극적으로 parameter 갯수 자체를 maximize 하는 것. (매우 효율적인 방식으로)
 - 이렇게 parameters 를 늘리지만, Floating point operations (FLOPs) 를 constant 값으로 바꾸는 것. 
    - 즉 parameters 는 늘리지만, 연산은 constant 하게 됨.  

<img src="{{ page.asset_path }}switch_transformer_model.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


## 기존 MOE 모델 방식

- MOE (Mixture of Experts) 개념은 이미 2017년 Shazeer et al에 의해 제안되었음. 
- route 에 해당하는 $$ W_r $$ 값이 존재하고, input값과 곱해져서 logits을 생성함 $$ h(x) = W_r \dot x $$ 
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
- route 에 해당하는 $$ W_r $$ 값이 존재하고, input값과 곱해져서 logits을 생성함 $$ h(x) = W_r \cdot x $$
- 이후에 softmax를 적용하고, top-k 개의 experts 를 선택함. 포인트는 여러개의 experts를 선택함

아래는 구체적인 공식

$$ h(x) = W_r \cdot x $$

- W_r : route matrix (이게 학습이 되면서 experts를 선택하는 역활을 함)
- x: input vector

이후에 softmax를 적용하여, top-k개의 experts를 선택함.

$$ p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}} $$

만약 T가 선택된 experts의 집합이라면, 최종 output은 다음과 같이 계산됩니다.

$$ y = \sum_{i \in T} p_i(x) E_i(x) $$

- E_i(x): 선택된 experts에서 i번째 expert의 output
- 그냥 곱하기 하면 됨

이걸 Pytorch 로 구현하면 다음과 같습니다. 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, k=2):
        super().__init__()
        self.k = k
        self.num_experts = num_experts

        # Router: logits = W_r · x
        self.router = nn.Linear(input_dim, num_experts)

        # Experts: each is a small MLP (or Linear here)
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        # Compute routing logits
        logits = self.router(x)  # (B, N)
        gate_probs = F.softmax(logits, dim=-1)  # (B, N)

        # Select top-k experts per example
        topk_vals, topk_idx = torch.topk(gate_probs, self.k, dim=-1)  # (B, k)

        # Initialize output
        output = torch.zeros(x.size(0), self.experts[0].out_features, device=x.device)

        for i in range(self.k):
            idx = topk_idx[:, i]  # expert index
            weight = topk_vals[:, i].unsqueeze(1)  # (B, 1)

            # For batch-wise selection
            expert_outputs = torch.stack([
                self.experts[expert_id](x[j].unsqueeze(0))
                for j, expert_id in enumerate(idx)
            ]).squeeze(1)  # (B, output_dim)

            output += weight * expert_outputs  # weighted sum

        return output
```


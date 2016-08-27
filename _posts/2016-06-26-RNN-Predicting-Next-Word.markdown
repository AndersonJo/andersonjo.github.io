---
layout: post
title:  "Recurrent Neural Network with TensorFlow"
date:   2016-06-25 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/RNN-With-Python/
tags: ['LSTM', 'tanh', 'hyperbolic tangent', 'cross-entropy loss', 'softmax']

---

<div>
    <img src="{{ page.asset_path }}brain.png" class="img-responsive img-rounded" style="width:100%">
</div>

### Recurrent Neural Network

$$ S_t = f(S_{t-1} * W + U_x * X_t) $$

| Variable | Description or Numpy Code |
|:---------|:------------|
| f | tanh, softmax 같은 함수 |
| $$ S_t $$          | 어느 한 시점 t 의 state |
| $$ S_{t-1} * W $$  | W.dot(S[t-1]) = (hidden, ) |
| $$ U_x * X_t $$    | U[:, x[t]] = (hidden, ) |
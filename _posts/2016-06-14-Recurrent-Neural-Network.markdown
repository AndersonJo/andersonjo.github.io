---
layout: post
title:  "TensorFlow - Recurrent Neural Network"
date:   2016-06-14 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Recurrent-Neural-Network/
tags: []

---

<div>
    <img src="{{ page.asset_path }}wallpaper-weird-self-esteem.jpg" class="img-responsive img-rounded">
</div>

# The Problem of RNN

RNN의 문제는 가장 최신의 이전 데이터를 통해서 현재를 볼 수 있지만.. 
아주 오래전 데이터 또는 Context를 통해서 현재를 보기 힘들다는 것입니다.

| O | The clouds are in the **sky** | context없이 sky를 추측할수 있습니다. |
| X | I grew up in France... I speak fluent **French** | 가장 최신 정보로는 어떤 language가 나오는 것은 알 수 있지만, French를 정확하게 알기 위해서는 France라는 Context를 알아야 합니다. |
 
즉 문제는, Gap이 커지면 커질수록 RNN은 정보를 연결시키지 못하는 단점을 갖고 있습니다. 
LSTM은 이 문제를 해결합니다.

# LSTM Networks

Long Short Term Memory Networks (LSTMs)는 long-term dependencies를 학습할수 있는 특수한 뉴럴 네크워크입니다.
일반적인 RNN은 다음과 같은 single tanh layer를 갖고 있는 형태를 띄고 있습니다.

<img src="{{ page.asset_path }}LSTM3-SimpleRNN.png" class="img-responsive img-rounded">

LSTM의 경우는 chain 같은 구조를 동일하게 갖고 있지만, 다른 repeating module 구조를 갖고 있습니다.

<img src="{{ page.asset_path }}LSTM3-chain.png" class="img-responsive img-rounded">

<img src="{{ page.asset_path }}LSTM2-notation.png" class="img-responsive img-rounded">
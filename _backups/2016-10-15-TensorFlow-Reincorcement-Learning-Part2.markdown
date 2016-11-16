---
layout: post
title:  "TensorFlow - Deep Reinforcement Learning Part 2"
date:   2016-10-15 01:00:00
categories: "tensorflow"
asset_path: /assets/posts2/TensorFlow/
tags: ['OpenAI', 'Neon', 'format']

---

<header>
    <img src="{{ page.asset_path }}google-breakout.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small> 
    </small>
    </div>
</header>

Part 2 에서는 DeepMind 팀에서 내놓은 Playing Atari with Deep Reinforcement Learning 논문을 해부할 것입니다.

### 1. Introduction

이미지 또는 음성등에서 바로 Agent를 학습시키는 것은 RL (Reinforcement Learning)에서 오래된 챌린지중의 하나입니다.<br>
이전의 RL방식들은 손으로 집접 만든 features들이나 policy등을 통해서 성공할수 있었지만 이 경우 특정 문제를 해결하는데만 최적화가 되어 있어 <br>
같은 방식으로 다른 문제들을 해결하기에는 어려운 점들이 많습니다.

최근 Deep Learning의 발전들은 raw sensory data (이미지등)에서 high-level features들을 뽑아내는게 가능하게 만들었고,<br>
이는 Convolutional networks, Multiplayer Perceptrons, restricted Boltzmann machines 그리고 
Recurrent Neural networks와 같은  컴퓨터 비전[11, 22, 16] 그리고 음성인식 [6, 7]에서의 비약적인 발전으로 이어졌습니다. 

하지만 Reinforcement Learning 은 deep learning의 관점에서 볼때 여러 챌린지들을 갖고 있습니다.<br>
첫번째로 성공적인 deep learning applications들은 수작업한 엄청나게 많은 데이터를 통해서 학습됩니다. 
하지만 RL 알고리즘은 그와는 반대로 scalar reward signal을 통해서 배워야만 하며, 
이 reward는 매우 적게 분포하고 있으며 (frequently sparse), delayed 된 경우가 많습니다.  
실질적으로 delay는 actions과 resulting rewards 사이에 수천 timesteps이 존재할정도로 거대합니다. 
이는 기존의 input과 targets이 direct로 연견될것과 비교해볼수 있습니다.<br>

다른 이슈는 기존의 대부분의 deep learning이 모든 samples들이 independent 하다고 여깁니다.<br> 
하지만 Deep Reinforcement Learning 에서는 매우 연관성이 높은 (correrated) states의 sequences를 만나게 될 일이 많습니다.

### Background

### 2. Background

environment $$ \epsilon $$에 해당하는 Atari emulator안에서 Agent는 일련의 actions, observations, 그리고 rewards등을 받습니다.<br>
각각의 time-step마다 Agent는 게임안에서 허용된 $$ A = \{1, ..., K\} $$ actions들로 부터 하나의 $$ a_{t} $$ (action)을 취하게 됩니다.<br>
Action은 environment (Atari Emulator)안으로 들어가게 되고, Game state, score가 변하게 됩니다. 
Emulator안의 내부 state자체를 Agent가 얻는것이 아니라 (예를 들어서 공 object의 위치나, paddle object의 위치 등등) 이미지 자체를 $$ x_t \in \Bbb{R}^d $$
(x 값이라는 것은.. input data로 사용된다는 뜻이고 x_t 라는건 어느 시점 (time)의 input image data를 말함) 

| Name | Math Symbol | Description |
|:-----|:------------|:------------|
| Environment | $$ \varepsilon $$ | Atari Emulator 를 뜻하며 Agent는 environment로 부터 actions, observations, rewards등을 주거니 받거니함<br>일반적으로  stochastic. |
| Action      | $$ a_t $$ | 특정 시점의 action을 말하며, 게임안에서 허용된 $$ A = \{1, ..., K\} $$ 중에 하나를 사용함 |
| Image (screen shot) | $$ x_t \in \Bbb{R}^d $$ | 현재시점의 화면을 나타내는 이미지 |
| Reward | $$ r_t $$ | Reward는 이전 **전체** actions 그리고 observations과 연관이 있음. <br>즉 하나의 action에 대한 feedback을 받으려면 일반적으로 수천번의 time-steps이 지나간 이후 받을수 있음  |   


### References 

* [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

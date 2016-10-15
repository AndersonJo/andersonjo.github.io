---
layout: post
title:  "TensorFlow - Deep Reinforcement Learning Part 1"
date:   2016-10-15 01:00:00
categories: "tensorflow"
asset_path: /assets/posts2/TensorFlow/
tags: ['game', 'deep Q-network']

---

<header>
    <img src="{{ page.asset_path }}google-breakout.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small>구글 이미지에서 Atari breakout이라고 치면 게임이 나옴. ㅎㄷㄷㄷㄷ        
    </small>
    </div>
</header>

# Reinforcement Learning 

<img src="{{ page.asset_path }}atari-breakout-deepmind.png" class="img-responsive img-rounded">

### Introduction

Atari Breakout을 학습시키기 위해서 Neural Network를 사용한다고 가정해봅시다. 
Input값은 현재 상태의 이미지이고, Output은 3개의 actions으로 귀속될수 있습니다. (왼쪽, 오른쪽, 그리고 공 던지기 - 시작할때)
Classification 문제가 될 것이고, 각각의 화면마다 왼쪽으로 옮길지, 오른쪽으로 옮길지 결정을 해야합니다.

문제는.. 이러한 방법은 엄청나게 많은 데이터셋 (전문가가 어떻게 어떻게 하는지 학습시켜줘야 함)을 요구합니다. <br>
인간은 그렇게 학습하지 않습니다. 우리 인간은 게임을 하기 위해서 누군가가 우리에서 와서 각각의 상황마다 우리한테 어떻게 하는지 일일이 다 말해주지 않습니다. 
보통 인간은 우리가 게임속에서 어떤 행동을 하게 되면, feedback을 받고 (죽거나 살거나 돈을 얻거나 레벨을 올리거나..) 어떻게 문제를 해결해 나갈지 알아냅니다.

**Reinforcement Learning**은 supervised와 unsupervised그 두 세계 사이에 위치해 있습니다.
Supervised learning에서는 각각의 모든 상황에 대해서 어떻게 해야할지 (target) 이미 정해놓은 상태이고, unsupervised에서는 어떻게 해야할지 전혀 데이터가 없는 상황입니다.
Reinforcement Learning에서는 **Sparse and Time-delayed labels**를 갖고 있습니다. 

### Credit Assignment Problem

저위의 간단한 게임이 어려운 이유는 공이 벽돌을 부시고 reward를 얻을때, 그 시점의 paddle의 위치 그리고 움직임은 해당 reward를 얻을때 기여한 바와 전혀 상관이 없습니다. 
즉 해당 reward를 얻기 위해서, 그 이전에 이미 paddle을 움직였고 공을 쳤기 때문에 그 다음 결과로서 reward가 발생한 것입니다.
이것을 credit assignment problem이라고 합니다. 어떤 이전의 actions이 reward를 받는데 일조했는지 그리고 어디까지 범위인지를 결정해야 하는 문제가 있습니다.


# Markov Decision Process

<img src="{{ page.asset_path }}drl-agent-environment.jpg" class="img-responsive img-rounded">

게임을 하는 사람을 **Agent**라고 하고, 게임을 **environment** (Breakout)라고 생각하면 됩니다.
Agent는 어떤 주어진 상태 **state** (paddle의 위치, 공의 위치와 방향, 모든 벽돌의 위치)안에서 
특정 (fixed) **actions** (paddle을 왼쪽으로 또는 오른쪽으로 움직이기)등을 할 수 있습니다. 
actions은 **reward**로 이어지기도 합니다. Actions은 environment를 변화 (transform) 시켜서 새로운 state로 이어지게 합니다
(새로운 state안에서 agent는 다시 새로운 actions을 취하겠죠) 
이러한 actions을 어떻게 취할 것인지 결정하는 것을 **Policy** 라고 부릅니다.
Environment는 일반적으로 **stochastic**입니다. 즉 다음 state는 random하게 변화될수 있다는 뜻입니다.
 

<img src="{{ page.asset_path }}mdp-illustrated.png" class="img-responsive img-rounded">

MDP는 간단하게 다음과 같이 쓸수 있습니다.

$$ \text{MDP} = \langle S,A,T,R,\gamma \rangle $$

| Sign | Definition | Description | 
|:-----|:-----------|:------------|
| S    | States  | [Grid Maps 예제](http://ais.informatik.uni-freiburg.de/teaching/ws12/mapping/pdf/slam11-gridmaps-4.pdf "Grid Map 예제") 참고. <br> BreakOut 에서는 paddle의 위치, 공의 위치과 방향, 모든 벽돌의 위치등등  |
| A    | Actions | Agent는 미리 정해둔(fixed) actions 들을 할 수 있습니다. 가령 paddle을 왼쪽, 오른쪽으로 움직이는 것들입니다. |
| T    | Transition Probability | 어떤 action을 했을때 어떤 한 state에서 다른 state로 넘어가는 확률입니다. | 
| R    | Reward | Breakout에서는 벽돌을 부시고 점수를 올리는게 reward이겠죠. |


### Discounted Future Reward

앤더슨이라는 개발자는 연봉으로 5,000만원을 받고 있습니다.<br>
현 상태가 계속 유지된다고 했을때, 평생동안 받게될 총 받게될 연봉액수는 다음과 같습니다. 

<div class="thumbnail" style="padding:0px 10px 0 10px;">
$$ 5000 + 5000 + 5000 + 5000 + ... = infinity $$ 
</div>


공식을 세우면 **Total Reward**는 다음과 같습니다.

<div class="thumbnail" style="padding:0px 10px 0 10px;">
$$ R = r_1 + r_2 + r_3 + ... + r_n $$
</div>


**Total Future Reward**는 다음과 같습니다.

<div class="thumbnail" style="padding:0px 10px 0 10px;">
$$ R_t = r_t + r_{t + 1} + r_{t+2} + ... + r_n $$
</div>

하지만 위의 계산에는 문제가 있습니다. 동일한 연봉으로 계속 받게 될 경우 미래에 받게 되는 연봉은 그 가치가 현재에 비해서 낮아질 것 입니다.
예를 들어서 인플레이션으로 인한 화폐 가치 하락으로 인하여, 10년뒤 받게될 연봉 5000만원은 지금의 5000만원의 가치에 비해서 매우 낮을 가능성이 있습니다. 

n years이후 받게되는 연봉의 가치가 $$ (0.9)^n * 5000 $$ 이라고 가정한다면 다음과 같이 쓸수 있습니다.

<div class="thumbnail" style="padding:0px 10px 0 10px;">
$$ 5000 + 0.9*5000 + 0.9^2 * 5000 + 0.9^3 * 5000 $$
</div>

**"Discounted sum of future rewards"** 는 **discount factor $$ \gamma $$** 를 사용해서 다음과 같이 나타낼수 있습니다.<br>
$$ \gamma $$ 는 gamma라고 읽으며 0 ~ 1 사이의 값을 갖습니다. n값은 특정 몇년후를 지정한 것입니다.

<div class="thumbnail" style="padding:0px 10px 0 10px;">
$$ R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... + \gamma^{n-t} r_n  $$
</div>

### References 

* [Guest Post (Part I): Demystifying Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
* [Intelligent Systems Lecture Notes](http://bluehawk.monmouth.edu/rclayton/web-pages/f11-520/mdp.html)
* [Carnegie Mellon University - Markov Decision Processes, and Dynamic Programming](http://www.cs.cmu.edu/~cga/ai-course/mdp.pdf)
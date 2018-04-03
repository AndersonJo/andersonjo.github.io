---
layout: post
title:  "TensorFlow - Deep Reinforcement Learning Part 1"
date:   2016-10-14 01:00:00
categories: "tensorflow"
asset_path: /assets/posts2/TensorFlow/
tags: ['game', 'deep Q-network']

---

<header>
    <img src="{{ page.asset_path }}google-breakout.jpg" class="img-responsive img-rounded img-fluid">
    <div style="text-align:right;"> 
    <small>구글 이미지에서 Atari breakout이라고 치면 게임이 나옴. ㅎㄷㄷㄷㄷ        
    </small>
    </div>
</header>

# Reinforcement Learning 

<img src="{{ page.asset_path }}atari-breakout-deepmind.png" class="img-responsive img-rounded img-fluid">

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

<img src="{{ page.asset_path }}drl-agent-environment.jpg" class="img-responsive img-rounded img-fluid">

게임을 하는 사람을 **Agent**라고 하고, 게임을 **environment** (Breakout)라고 생각하면 됩니다.
Agent는 어떤 주어진 상태 **state** (paddle의 위치, 공의 위치와 방향, 모든 벽돌의 위치)안에서 
특정 (fixed) **actions** (paddle을 왼쪽으로 또는 오른쪽으로 움직이기)등을 할 수 있습니다. 
actions은 **reward**로 이어지기도 합니다. Actions은 environment를 변화 (transform) 시켜서 새로운 state로 이어지게 합니다
(새로운 state안에서 agent는 다시 새로운 actions을 취하겠죠) 
이러한 actions을 어떻게 취할 것인지 결정하는 것을 **Policy** 라고 부릅니다.
Environment는 일반적으로 **stochastic**입니다. 즉 다음 state는 random하게 변화될수 있다는 뜻입니다.
 

<img src="{{ page.asset_path }}mdp-illustrated.png" class="img-responsive img-rounded img-fluid">

MDP는 간단하게 다음과 같이 쓸수 있습니다.

$$ \text{MDP} = \langle S,A,T,R,\gamma \rangle $$

| Sign | Definition | Description | 
|:-----|:-----------|:------------|
| S    | States  | [Grid Maps 예제](http://ais.informatik.uni-freiburg.de/teaching/ws12/mapping/pdf/slam11-gridmaps-4.pdf "Grid Map 예제") 참고. <br> BreakOut 에서는 paddle의 위치, 공의 위치과 방향, 모든 벽돌의 위치등등  |
| A    | Actions | Agent는 미리 정해둔(fixed) actions 들을 할 수 있습니다. 가령 paddle을 왼쪽, 오른쪽으로 움직이는 것들입니다. |
| T    | Transition Probability | 어떤 action을 했을때 어떤 한 state에서 다른 state로 넘어가는 확률입니다. | 
| R    | Reward | Breakout에서는 벽돌을 부시고 점수를 올리는게 reward이겠죠. |


## Discounted Future Reward

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
$$ \gamma $$ 는 gamma라고 읽으며 0 ~ 1 사이의 값을 갖습니다. n값은 특정 몇년후를 지정한 것이고, t는 time point입니다. 즉...<br>
$$ r_t $$ 는 t라는 시점에 받은 reward를 뜻하며, $$ \gamma \in ]0, 1[ $$ 입니다. 


<div class="thumbnail" style="padding:0px 10px 0 10px;">
$$ R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... + \gamma^{n-t} r_n = \sum_{t=0}{ \gamma^t r_t}  $$

$$ R_t = r_t + \gamma( r_{t+1} + \gamma (r_{t+2} + ... )) = r_t + \gamma R_{t+1} $$
</div>

$$ r_t + \gamma R_{t+1} $$ 이말은.. 현재의 reward + 그 다음에 올 모든 rewards를 뜻합니다.<br>
목표는 discounted sum of future rewards가 maximize시키는 것입니다. 

## Real Life of Developer

<img src="{{ page.asset_path }}real-life-of-developer.png" class="img-responsive img-rounded img-fluid">

아마 현실세계에서는 좀 더 복잡할 것입니다. 다음과 같이 정의 해보겠습니다.

* $$ J_I $$ = 인턴 개발자로 상태(state)로 시작했을때의 Expected discounted future rewards
* $$ J_J $$ = 쥬니어 개발자의 상태(state)로 시작했을때의 Expected discounted future rewards
* $$ J_S $$ = 시니어 개발자의 상태(state)로 시작했을때의 Expected discounted future rewards
* $$ J_F $$ = 백수의 상태(state)로 시작했을때의 Expected discounted future rewards
* $$ J_D $$ = 죽었을때의 상태(state)로 시작했을때의 Expected discounted future rewards

문제는 어떻게 $$ J_I, J_J, J_S, J_F, J_D $$를 연산해서 구하는 것입니다.

# Q-Learning by Example

예를 통해서 Q-Learning을 배워보도록 하겠습니다. 먼저 **Q matrix 를 0값으로 initialize**해줍니다. (보통 일반적으로 0으로 초기화 해줍니다.)<br>
**Gamma 값은 0.8**로 잡겠습니다.

<img src="{{ page.asset_path }}Q-Learning-Example_clip_image004.gif" class="img-responsive img-rounded img-fluid">

Reward Matrix **R**의 State B를 보면 2개의 possible actions이 있습니다. 하나는 state D로 가는 것이고, 다른 하나는 state F로 가는 것입니다. 
Random Selection에 따라, state F를 따라갑니다. state F에는 3개의 possible actions이 있습니다. B E 또는 F
 
 <img src="{{ page.asset_path }}Q-Learning-Example_clip_image006.gif" class="img-responsive img-rounded img-fluid">

<div class="thumbnail" style="padding:0px 10px 0 10px;">
$$ \mathbf{Q}(state, action) = \mathbf{R}(state, action) + \gamma \cdot \mathbf{Max}[\mathbf{Q}(next \ state, all \ actions)] $$

$$ \mathbf{Q}(B, F) = \mathbf{R}(B, F) + 0.8 \cdot \mathbf{Max} \{ \mathbf{Q}(F, B),\ \mathbf{Q}(F,E),\ \mathbf{Q}(F,F)\} = 100 + 0.8 \cdot 0 = 100 $$
</div>

여기에서 matrix Q는 현재 모든 값이 0이기 때문에, $$ \mathbf{Q}(F, B),\ \mathbf{Q}(F,E),\ \mathbf{Q}(F,F) $$ 들은 모두 0 입니다. <br>
instant reward 때문에, $$ \mathbf{Q}(B, F) $$ 의 결과값은 100입니다.

자! 이제 F가 현재의 state가 되었습니다. F가 최종 목표 state이기 때문에 여기에서 1 episode를 마칩니다. <br>
마쳐진이후, Agent의 brain는 (matrix Q) 다음과 같이 업데이트 됩니다.
 
<img src="{{ page.asset_path }}Q-Learning-Example_clip_image014.gif" class="img-responsive img-rounded img-fluid">

그 다음 episode로, 이번에는 state D에서 시작을 합니다. B, C, E로 갈수 있는데 모두 동일한 0값을 갖고 있고, random selection에 의해서 B로 갑니다.
B로 가게 되면 2 possible actions이 있고, D 또는 F로 갈 수 있습니다. 

<div class="thumbnail" style="padding:0px 10px 0 10px;">
$$ \mathbf{Q}(state, action) = \mathbf{R}(state, action) + \gamma \cdot \mathbf{Max}[\mathbf{Q}(next \ state, all \ actions)] $$

$$ \mathbf{Q}(D, B) = \mathbf{R}(D, B) + 0.8 \cdot \mathbf{Max} \{ \mathbf{Q}(B, D),\ \mathbf{Q}(B,F)\} = 100 + 0.8 \cdot Max\{0,\ 100\} = 80 $$
</div>

Q matrix는 다음과 같이 업데이트 됩니다.

<img src="{{ page.asset_path }}Q-Learning-Example_clip_image024.gif" class="img-responsive img-rounded img-fluid">

이런 식으로 계속 반복해서 훈련을 시키다 보면 다음과 같은 Q matrix가 만들어집니다.
 
<img src="{{ page.asset_path }}Q-Learning-Example_clip_image030.gif" class="img-responsive img-rounded img-fluid">
 
수학적으로 정리하면 다음과 같이 될 수 있습니다.

<div class="thumbnail" style="padding:0px 10px 0 10px;">
$$ \mathbf{Q}(state, action) = r + \gamma \cdot argmax\{ \mathbf{Q}(next\ state, next\ action) \} $$
</div>

# Deep Q Network

Atary Breakout게임을 Q Learning으로 만들기 위해서 states는 paddle의 위치, 공의 위치와 방향, 각각의 벽돌의 존재등등.. 
하지만 이런 특정 게임의 속성 값들을 state로 쓰면 다른 게임에 적용할수가 없게 됩니다. 

DeepMind paper에 따르면 게임 스크린 자체의 픽셀들을 state로 사용할 수 있습니다. 
문제는 공의 방향인데 (스크린 하나만 놓고 보면.. 알수가 없음..), DeepMind에서는 4개의 스크린샷을 뜨고, 84*84 의 256 gray색상을 갖은 이미지로 리싸이즈 했습니다.
즉 $$ 256^{84 * 84 * 4} = 10^{67970} $$ 이고.. 한마디로 Q Table안에 $$ 10^{67970} $$ 개의 rows들을 갖어야 합니다. (알려진 원자의 갯수보다도 더 많습니다. ㅡㅡ)
설령 만들어서 한다고 해도, 학습시키는데.. 제가 죽기전에 끝날지도 모를것입니다. 

여기에서 Deep Q Network가 등장합니다.<br>
Q-function 자체를 neural network로 만들면 문제가 해결될 수 있습니다. 즉 states는 4장의 게임 화면으로, actions은 동일하게 가져가고, output으로는 Q-value를 내놓습니다.
다른 방법으로는, 4장의 게임 화면만 input으로 받고, output으로 각각의 action에 따른 Q-value를 내놓게 할 수 도 있습니다. 

<img src="{{ page.asset_path }}deep-q-learning-001.png" class="img-responsive img-rounded img-fluid">

DeepMind에서 사용한 network architecture는 다음과 같습니다.

<img src="{{ page.asset_path }}deep-q-learning-used-by-deepmind.png" class="img-responsive img-rounded img-fluid">

3개의 convolutional layers를 갖고 있는 일반적인 convolutional neural network입니다. 재미있는건 pooling layers가 없습니다. 
pooling layers 사용시, object의 위치가 불명확해집니다. 이런 경우 ImageNet같은 classification task에는 맞습니다. 
하지만 object의 위치를 정확하게 알아야 하는 게임에서는 위치를 모를경우 reward를 주거나 하는 방법이 없어져 버립니다.

Input값은 4장의 84 * 84 grayscale 게임 스크린이 됩니다. 
Output은 각각의 actions에 따른 Q-values가 됩니다. Q-values는 real values가 될 수 있고.. 이러한 특징은 squared error loss로 optimize할수 있는 regression task만들수 있습니다.
 
<div class="thumbnail" style="padding:0px 10px 0 10px;">
$$ \mathbf{L} = \frac{1}{2}\{ \mathbf{r} + max[ \mathbf{Q}(next\ state, next\ action)] - \mathbf{Q}(state, action) \}^2 $$
</div>
 

### References 

* [Guest Post (Part I): Demystifying Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
* [Intelligent Systems Lecture Notes](http://bluehawk.monmouth.edu/rclayton/web-pages/f11-520/mdp.html)
* [Carnegie Mellon University - Markov Decision Processes, and Dynamic Programming](http://www.cs.cmu.edu/~cga/ai-course/mdp.pdf)
* [Northeastern University - Markov Decision Processes](http://www.ccs.neu.edu/home/rplatt/cs5100_2015/slides/mdps.pdf)
* [STUDYWOLF - REINFORCEMENT LEARNING PART 1: Q-LEARNING AND EXPLORATION](https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/)
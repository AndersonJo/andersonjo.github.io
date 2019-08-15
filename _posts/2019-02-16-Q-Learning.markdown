---
layout: post
title:  "Q-Learning for Mountain Car"
date:   2019-02-10 01:00:00
categories: "machine-learning"
asset_path: /assets/images/
tags: ['mcts', 'taxi', 'tictactoe', 'tic-tac-toe', 'othello', 'alphazero', 'alphago']
---



<header>
    <img src="{{ page.asset_path }}q-learning.jpeg" class="img-responsive img-rounded img-fluid">
    <div style="text-align:right;">
    
    </div>
</header>


# Mountain Car 

목표는 언덕위로 차량을 올려놓는 것 입니다. 

### 학습 완료된 화면

<img src="{{ page.asset_path }}mountain-car.gif" class="img-responsive img-rounded img-fluid">


### Observation

| Num | Observation | Min | Max |
|:----|:------------|:----|:----|
| 0   | position    | -1.2 | 0.6 |
| 1   | velocity    | -0.07 | 0.07 |

{% highlight python %}
env = gym.make('MountainCar-v0')
env.observation_space.high  # array([0.6 , 0.07], dtype=float32)
env.observation_space.low   # array([-1.2 , -0.07], dtype=float32)
{% endhighlight %}

### Actions


| Num | Action     |
:-----|:-----------|
| 0   | push left  |
| 1   | no push    |
| 2   | push right |

# Q-Learning


## Bellman Equation 

$$ Q(s, a) = learning\ rate \cdot (r + \gamma( max(Q(s^{\prime}, a^{\prime})))) $$

## Q Function

$$ Q(s, a) = Q(s,a) + \text{lr} \left[ R(s, a) + \gamma \max Q^\prime (s^\prime, a^\prime) - Q(s, a) \right] $$

* $$ \text{lr} $$ : Learning rate
* $$ R(s, a) $$ : 현재 state, action으로 얻은 reward
* $$ Q $$ : 현재의 Q value
* $$ \max Q^\prime (s^\prime, a^\prime) $$ : Maximum future reward
* $$ s^\prime $$ : step(action)으로 얻은 next_state
* $$ \gamma $$ : Discount rate


## Build Q Table

Continuous value를 어떻게든 테이블로 만들도록 잘라넣습니다. 

{% highlight python %}
env = gym.make('MountainCar-v0')
n_state = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
n_state = np.round(n_state, 0).astype(int) + 1

Q = np.random.uniform(-1, 1, size=(n_state[0], n_state[1], env.action_space.n))
print('Q shape:', Q.shape)
print('Q Table')
print(Q[1:2])
{% endhighlight %}

{% highlight python %}
Q shape: (19, 15, 3)
Q Table
[[[ 0.99048129  0.83269269  0.23944522]
  [-0.4517455  -0.76882655 -0.00480888]
  [ 0.61718192  0.01420441 -0.08474976]
  [-0.38611008  0.34376222 -0.71499911]
  [-0.78333052 -0.30410788 -0.30258901]
  [ 0.8138172  -0.69035782  0.99421675]
  [-0.73070808 -0.60350616  0.57929507]
  [ 0.81467379 -0.82851229 -0.44759567]
  [-0.83048389  0.00949504 -0.28621805]
  [-0.80981087 -0.54730307  0.39901784]
  [-0.98453426  0.12534842  0.4347526 ]
  [-0.51690061 -0.69667071 -0.13774189]
  [ 0.91651489 -0.88653031 -0.93615038]
  [ 0.0208071   0.19121545 -0.32631843]
  [ 0.34336055  0.10997157  0.60867634]]]
{% endhighlight %}


## training

{% highlight python %}
def discretize(env, state):
    state = (state - env.observation_space.low) * np.array([10, 100])
    state = np.round(state, 0).astype(int)
    return state

def train(env, Q, epochs=10000, lr=0.1, gamma=0.9, epsilon=0.9):
    reduction = epsilon/epochs
    action_n = env.action_space.n
    
    rewards = list()
    
    for epoch in tqdm_notebook(range(epochs)):
        state = env.reset()
        state = discretize(env, state)
        
        done = False
        _tot_reward = 0
        _tot_rand_action = 0
        _tot_q_action = 0
        _max_pos = 0
        
        while not done:

            # Calculate next action
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state[0], state[1]])
                _tot_q_action += 1
            else:
                action = np.random.randint(0, action_n)
                _tot_rand_action += 1
                
            # Step!
            next_state, reward, done, info = env.step(action)
            next_state_apx = discretize(env, next_state)

            # Terminal Update
            if done and next_state[0] >= 0.5:
                Q[next_state_apx[0], next_state_apx[1], action] = reward
            else:
                delta = lr * (reward + gamma * np.max(Q[next_state_apx[0], next_state_apx[1]]) - 
                              Q[state[0], state[1], action])
                Q[state[0], state[1], action] += delta
            
            state = next_state_apx
            _tot_reward += reward
            
        # Decay Epsilon
        if epsilon > 0:
            epsilon -= reduction
            epsilon = round(epsilon, 4)
            
        # Track Rewards
        rewards.append(_tot_reward)
        
        # Log
        if epoch%100 == 0:
            print(f'\repoch:{epoch} | tot reward:{_tot_reward} | epsilon:{epsilon} | ' 
                  f'rand action:{_tot_rand_action} | Q action:{_tot_q_action}')

train(env, Q)
{% endhighlight %}


{% highlight bash %}
epoch:0 | tot reward:-200.0 | epsilon:0.8999 | rand action:178 | Q action:22
epoch:100 | tot reward:-200.0 | epsilon:0.8899 | rand action:170 | Q action:30
epoch:200 | tot reward:-200.0 | epsilon:0.8799 | rand action:168 | Q action:32
epoch:300 | tot reward:-200.0 | epsilon:0.8699 | rand action:170 | Q action:30
epoch:400 | tot reward:-200.0 | epsilon:0.8599 | rand action:163 | Q action:37
epoch:500 | tot reward:-200.0 | epsilon:0.8499 | rand action:164 | Q action:36
epoch:600 | tot reward:-200.0 | epsilon:0.8399 | rand action:165 | Q action:35
epoch:700 | tot reward:-200.0 | epsilon:0.8299 | rand action:162 | Q action:38
epoch:800 | tot reward:-200.0 | epsilon:0.8199 | rand action:159 | Q action:41
epoch:900 | tot reward:-200.0 | epsilon:0.8099 | rand action:155 | Q action:45
epoch:1000 | tot reward:-200.0 | epsilon:0.7999 | rand action:162 | Q action:38
epoch:1100 | tot reward:-200.0 | epsilon:0.7899 | rand action:163 | Q action:37
epoch:1200 | tot reward:-200.0 | epsilon:0.7799 | rand action:150 | Q action:50
epoch:1300 | tot reward:-200.0 | epsilon:0.7699 | rand action:139 | Q action:61
epoch:1400 | tot reward:-200.0 | epsilon:0.7599 | rand action:155 | Q action:45
epoch:1500 | tot reward:-200.0 | epsilon:0.7499 | rand action:148 | Q action:52
epoch:1600 | tot reward:-200.0 | epsilon:0.7399 | rand action:148 | Q action:52
epoch:1700 | tot reward:-200.0 | epsilon:0.7299 | rand action:146 | Q action:54
epoch:1800 | tot reward:-200.0 | epsilon:0.7199 | rand action:139 | Q action:61
epoch:1900 | tot reward:-200.0 | epsilon:0.7099 | rand action:149 | Q action:51
epoch:2000 | tot reward:-200.0 | epsilon:0.6999 | rand action:141 | Q action:59
epoch:2100 | tot reward:-200.0 | epsilon:0.6899 | rand action:144 | Q action:56
epoch:2200 | tot reward:-200.0 | epsilon:0.6799 | rand action:130 | Q action:70
epoch:2300 | tot reward:-200.0 | epsilon:0.6699 | rand action:121 | Q action:79
epoch:2400 | tot reward:-200.0 | epsilon:0.6599 | rand action:134 | Q action:66
epoch:2500 | tot reward:-200.0 | epsilon:0.6499 | rand action:112 | Q action:88
epoch:2600 | tot reward:-200.0 | epsilon:0.6399 | rand action:135 | Q action:65
epoch:2700 | tot reward:-200.0 | epsilon:0.6299 | rand action:124 | Q action:76
epoch:2800 | tot reward:-200.0 | epsilon:0.6199 | rand action:123 | Q action:77
epoch:2900 | tot reward:-200.0 | epsilon:0.6099 | rand action:123 | Q action:77
epoch:3000 | tot reward:-200.0 | epsilon:0.5999 | rand action:126 | Q action:74
epoch:3100 | tot reward:-200.0 | epsilon:0.5899 | rand action:109 | Q action:91
epoch:3200 | tot reward:-200.0 | epsilon:0.5799 | rand action:124 | Q action:76
epoch:3300 | tot reward:-200.0 | epsilon:0.5699 | rand action:114 | Q action:86
epoch:3400 | tot reward:-200.0 | epsilon:0.5599 | rand action:103 | Q action:97
epoch:3500 | tot reward:-200.0 | epsilon:0.5499 | rand action:115 | Q action:85
epoch:3600 | tot reward:-200.0 | epsilon:0.5399 | rand action:99 | Q action:101
epoch:3700 | tot reward:-200.0 | epsilon:0.5299 | rand action:118 | Q action:82
epoch:3800 | tot reward:-200.0 | epsilon:0.5199 | rand action:106 | Q action:94
epoch:3900 | tot reward:-200.0 | epsilon:0.5099 | rand action:97 | Q action:103
epoch:4000 | tot reward:-200.0 | epsilon:0.4999 | rand action:108 | Q action:92
epoch:4100 | tot reward:-200.0 | epsilon:0.4899 | rand action:106 | Q action:94
epoch:4200 | tot reward:-200.0 | epsilon:0.4799 | rand action:91 | Q action:109
epoch:4300 | tot reward:-200.0 | epsilon:0.4699 | rand action:84 | Q action:116
epoch:4400 | tot reward:-198.0 | epsilon:0.4599 | rand action:76 | Q action:122
epoch:4500 | tot reward:-200.0 | epsilon:0.4499 | rand action:92 | Q action:108
epoch:4600 | tot reward:-200.0 | epsilon:0.4399 | rand action:91 | Q action:109
epoch:4700 | tot reward:-200.0 | epsilon:0.4299 | rand action:83 | Q action:117
epoch:4800 | tot reward:-200.0 | epsilon:0.4199 | rand action:75 | Q action:125
epoch:4900 | tot reward:-200.0 | epsilon:0.4099 | rand action:88 | Q action:112
epoch:5000 | tot reward:-200.0 | epsilon:0.3999 | rand action:84 | Q action:116
epoch:5100 | tot reward:-200.0 | epsilon:0.3899 | rand action:76 | Q action:124
epoch:5200 | tot reward:-200.0 | epsilon:0.3799 | rand action:71 | Q action:129
epoch:5300 | tot reward:-200.0 | epsilon:0.3699 | rand action:68 | Q action:132
epoch:5400 | tot reward:-200.0 | epsilon:0.3599 | rand action:75 | Q action:125
epoch:5500 | tot reward:-200.0 | epsilon:0.3499 | rand action:64 | Q action:136
epoch:5600 | tot reward:-200.0 | epsilon:0.3399 | rand action:72 | Q action:128
epoch:5700 | tot reward:-200.0 | epsilon:0.3299 | rand action:79 | Q action:121
epoch:5800 | tot reward:-200.0 | epsilon:0.3199 | rand action:68 | Q action:132
epoch:5900 | tot reward:-200.0 | epsilon:0.3099 | rand action:72 | Q action:128
epoch:6000 | tot reward:-200.0 | epsilon:0.2999 | rand action:57 | Q action:143
epoch:6100 | tot reward:-200.0 | epsilon:0.2899 | rand action:70 | Q action:130
epoch:6200 | tot reward:-200.0 | epsilon:0.2799 | rand action:48 | Q action:152
epoch:6300 | tot reward:-200.0 | epsilon:0.2699 | rand action:51 | Q action:149
epoch:6400 | tot reward:-200.0 | epsilon:0.2599 | rand action:54 | Q action:146
epoch:6500 | tot reward:-200.0 | epsilon:0.2499 | rand action:34 | Q action:166
epoch:6600 | tot reward:-200.0 | epsilon:0.2399 | rand action:56 | Q action:144
epoch:6700 | tot reward:-158.0 | epsilon:0.2299 | rand action:38 | Q action:120
epoch:6800 | tot reward:-200.0 | epsilon:0.2199 | rand action:39 | Q action:161
epoch:6900 | tot reward:-190.0 | epsilon:0.2099 | rand action:33 | Q action:157
epoch:7000 | tot reward:-200.0 | epsilon:0.1999 | rand action:41 | Q action:159
epoch:7100 | tot reward:-200.0 | epsilon:0.1899 | rand action:40 | Q action:160
epoch:7200 | tot reward:-161.0 | epsilon:0.1799 | rand action:27 | Q action:134
epoch:7300 | tot reward:-200.0 | epsilon:0.1699 | rand action:26 | Q action:174
epoch:7400 | tot reward:-200.0 | epsilon:0.1599 | rand action:36 | Q action:164
epoch:7500 | tot reward:-159.0 | epsilon:0.1499 | rand action:26 | Q action:133
epoch:7600 | tot reward:-159.0 | epsilon:0.1399 | rand action:21 | Q action:138
epoch:7700 | tot reward:-158.0 | epsilon:0.1299 | rand action:13 | Q action:145
{% endhighlight %}


## Playing

{% highlight python %}
env = gym.make('MountainCar-v0')
state = env.reset()
state = discretize(env, state)

env.render()
input()

while True:
    env.render()
    action = np.argmax(Q[state[0], state[1]])
    state, reward, done, info = env.step(action)
    state = discretize(env, state)
    
    print(f'\rstate:{state} | reward:{reward} | done:{done} | info:{info}')
    
    if done:
        break
        
{% endhighlight %}
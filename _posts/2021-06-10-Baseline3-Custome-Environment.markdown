---
layout: post
title:  "Baseline3 - GYM Custom Environment"
date:   2021-06-10 01:00:00
categories: "reinforcement-learning"
asset_path: /assets/images/
tags: ['baseline3']
---

# Environment 101 

## Action or Observation Spaces 

1. **gym.spaces.Discrete**
   - 한번에 하나의 액션을 취할때 사용 
   - range: [0, n-1]
   - Discrete(3) 의경우 0, 1, 2 의 액션이 존재 

2. **gym.spaces.MultiDiscrete**
   - Discrete 의 묶음이라고 보면 됨
   - 예를 들어, 방향키 + 액션버튼1개는 `MultiDescrete([5, 2])` 이렇게 표현됨
      - Arrow Keys: `Discrete(5)` -> NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]
      - Action A: `Discrete(2)` -> NOOP[0], Action A[1]
      

3. **gym.spaces.Box**
   - 다차원 공간의 matrix 또는 tensor 형태의 데이터를  표현할때 사용
   - parameters
      - low: random sampling시에 minimum value
      - high: random sampling시에 maximum value 
   - ```python
     box = gym.spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
     box.sample()
     array([1.76413282, 0.48308708])
     ```
   - ```python
     box = gym.spaces.Box(low=-1, high=1, shape=(3, 2), dtype=np.float32)
     box.sample()
     array([[ 0.6516619 , -0.01624624],
            [ 0.48928288, -0.8799452 ],
            [ 0.73846203,  0.7973261 ]], dtype=float32)
     ```
     
2. **gym.spaces.Dict**
    - dictionary 로 데이터를 표현할때 사용
    - ```python
      from gym import spaces
      observation = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)}) 
      ```
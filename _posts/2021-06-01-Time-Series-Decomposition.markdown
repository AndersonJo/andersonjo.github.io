---
layout: post
title:  "Time-Series Decomposition"
date:   2021-06-01 01:00:00
categories: "time-series"
asset_path: /assets/images/
tags: ['']
---

# Introduction

Time series decomposition은 복잡한 시계열 데이터를 각각의 컴포넌트 파트로 쪼개는 과정입니다. <br> 
수리적으로 표현하면 다음과 같습니다. 

**Additive Decomposition**

$$ y_t = S_t + T_t + C_t + R_t $$

**Multiplicative Decomposition**

$$ \begin{align} y_t &= S_t \times T_t \times C_t  \times R_t \\
\log y_t &= \log S_t + \log T_t \times \log C_t \times \log R_t
\end{align} $$


- $$ y_t $$ : 데이터
- $$ S_t $$ : **S**easonal component
- $$ T_t $$ : **T**rend-cycle component 
- $$ C_t $$ : **C**yclical component (보통 Cyclical 그리고 Remainder를 합쳐서 Noise 로 보기도 함)
- $$ R_t $$ : **R**emainder component



1. 이게 무슨 의미? 
    - 시계열 데이터로 예측할때 component를 나눈후 따로따로 데이터를 취급하여 예측력을 높임
2. Additive Decomposition  특징
    - 선형 모델링. 변화가 지속적으로 고르게 일어남. 
3. Multiplicative Decomposition 특징
    - 비선형 모델링. Quadratice 또는 exponential 변화가 일어나는 곳에 쓰임. 
    - 주식은 비선형에 가까움
4. 기타
    - 두개의 decompositions 을 합쳐서 사용하기도 함
    - 현실에서는 그냥 한번에 끝나지 않을수도 있음. <br>decomposition을 하기전에 많은 데이터 전처리가 들어갈수도 있음
   

# Classical Decomposition

1. **Seasonal Period**
    - 분기 데이터: `m = 4` (1분기, 2분기,.. 4분기)
    - 월별 데이터: `m = 12` (1월, 2월.. 12월)
    - 일별 데이터: `m =7` (월, 화, ... 일)

Classical decomposition에서는 **seasonal period값이 상수값**으로 사용합니다. <br>
즉 주가처럼 변화가 심한 것은 다른 알고리즘이 맞음


## Additive Decomposition 

1. Trend Cycle 을 계산
    - 공식: $$ \hat{T}_t =  MA(m) $$
    - 원래 공식은 m이 odd 또는 even이냐에 따라 공식이 달라지는데 그냥 MA구한다고 생각하면 됨 
2. Detrended Series 계산 : 
    - Detranded Series = $$ y_t - \hat{T}_t $$
3. Seasonal Component 계산 
    - 예를 들어 월별 데이터에서 3월의 seasonal 값을 구한다면.. 모든 3월 데이터의 평균값을 구합니다. 
    - 예) (2019년 3월 + 2020년 3월 + 2021년 3월)/3 = averaged seasonality 
4. Remainder Component 계산
    - 공식: $$ R_t = y_t - \hat{T}_t - \hat{S}_t $$
    
* $$ MA(m) $$ : Moving average of order m -> MA(4) 면 df.rolling(4).mean() 과 같음


## Multiplicative Decomposition

Additive 와 매우 유사합니다. 

1. MA계산. 
2. Detrended Series = $$ y_t\ / \ \hat{T}_t $$
3. Additve decomposition과 동일
4. 공식: $$ R_t = y_t \big/ \left(\hat{T}_t \times \hat{S}_t \right) $$


## 의견

1. Classical decomposition은 자주 사용되나 추천은 안함. 이유.. 
    1. MA를 구하면서 앞, 뒤 데이터가 잘려나감 
    2. Seasnal change에 약함. <br>예를들어 60년대 전기 사용이 겨울철이 많았다면, <br>현재는 에어컨 사용으로 여름에 더 전기 사용량이 많음. <br>-> classical decomposition이 이런 변화에 약함
    3. 항공 산업 전체에서 일시적인 파업등으로 일부 데이터가 잠시 변화가 있을시.. 여기에 robust하게 대처 못함


## Python Code


{% highlight python %}
import pandas as pd
import kaggle.api as kaggle
from tempfile import gettempdir
from pathlib import Path

data_path = Path(gettempdir()) / 'AirPassengers.csv'

kaggle.authenticate()
kaggle.dataset_download_files('rakannimer/air-passengers', data_path.parent, unzip=True)
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index)
{% endhighlight %}


{% highlight python %}
from statsmodels.tsa.seasonal import seasonal_decompose

# 월별 데이터라서  period=12 로 잡음. 값 안넣어도 자동으로 12로 잡힘
# model="multiplicative" 넣으면 multiplicative decomposition 함
dec = seasonal_decompose(df, model='additive', period=12)
fig = dec.plot()
fig.set_size_inches(9, 5)
{% endhighlight %}

<img src="{{ page.asset_path }}decomposition01.png" class="img-responsive img-rounded img-fluid border rounded">
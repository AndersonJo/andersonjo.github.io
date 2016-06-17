---
layout: post
title:  "TensorFlow - Time Series Neural Network"
date:   2016-05-10 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/TS-Neural-Network/
tags: ['Time-Series']

---

<div>
    <img src="{{ page.asset_path }}fantasy-time-clock.jpg" class="img-responsive img-rounded">
</div>

# Simple Explanation

예를 들어서 수년동안 월 단위로 축적된 데이터 36개가 있고, 1개월 뒤를 미리 예측하고 싶을때.. 

1. Exploratory Data Analysis: 일단 전통적인 시계열 분석 방법을 적용하여 lag dependence를 알아냅니다. 
   (Auto correlation, partial auto correlation plots, transformations, differencing)
   즉 특정 월의 값이 이전 3개원 전의 데이터들과 관련이 있는지 없는지를 분석합니다.
2. training 24개, validation용으로 12개로 나눕니다. 
3. Neural Net을 만듭니다. 이전 3개월치의 데이터를 input 값으로 받고, 이를 토대로 다음 1개월 뒤를 예측하고자 합니다. 
4. 학습 시킵니다. 각각의 트레이닝 패턴은 4개의 데이터가 됩니다. (마지막은 데이터는 학습을 위한 실제 값)<br>
    $$ x_{1}, x_{2}, ..., x_{24}  $$ 가 있을때..<br> 
    $$ x_{1}, x_{2}, x_{3}, x_{4}  $$<br>
    $$ x_{2}, x_{3}, x_{4}, x_{5}  $$<br>
    ...<br>
    $$ x_{21}, x_{22}, x_{23}, x_{24}  $$
5. 테스트를 진행합니다. (25~35개월 데이터를 갖고서 잘 맞는지 확인 합니다.)
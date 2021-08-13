---
layout: post 
title:  "Ensemble / Bias and Variance Tradeoff / Bagging VS Boosting"
date:   2015-10-25 01:00:00 categories: "machine-learning"
asset_path: /assets/images/ 
tags: ['underfitting', 'overfitting', '오버피팅']

---

# Bias and Variance Tradeoff

<img src="{{ page.asset_path }}underfit_right_overfit.png" class="img-responsive img-rounded img-fluid center">

# Error

$$ Error = noise(x) + bias(x) + variance(x) $$

- error: irreducible error 로서 제거 할수 없는 존재..
- bias 그리고 variance: 서로 tradeoff 관계에 있으며 적절하게 조정해서 minimize 하는게 목적



## Bias 

Bias는 예측된 값과 기대값(GT) 차이라고 볼 수 있습니다. 


$$ \text{Bias} = E[y - \hat{y}] $$

 - Bias는 예측관 값과 Ground Truth 값과의 차이의 평균입니다.
 - Low Bias: 약한 가정 / 오차가 적다 (Decision Tree, KNN, SVM)
 - High Bias: 강한 가정 / 오차가 크다 (Linear Regression, Linear Discriminant Analysis, Logistic Regression)


## Variance 

$$ \text{Variance} = E \left[ \hat{y} - E[\hat{y}] \right]^2  $$

 - 예측값과 예측값들의 평균의 차를 제곱해준 것입니다. 
 - 즉 예측값들끼리 얼마나 퍼져 있는지, 좁게 몰려 있는지를 나타낸 것입니다.
 - Low Variance: 
 - High Variance: 모든 데이터들을 지나치게 학습
 
<img src="{{ page.asset_path }}bias_variance_tradeoff.jpeg" class="img-responsive img-rounded img-fluid center">


## Underfitting and Overfitting

 - Underfitting : High Bias and Low Variance
     - 모델은 예측시 강한 가정(assumption)을 갖고 있음. 
     - 데이터 부족으로 정확한 모델을 만들 수 없을 때 발생
     - Linear Model 을 Non-Linear Data 에 적용할때 발생
 - Overfitting : Low Bias and High Variance 
     - 노이즈 데이터까지 피팅 시켜서 발생함
     - 복잡한 모델을 단순한 데이터에 적용시 발생
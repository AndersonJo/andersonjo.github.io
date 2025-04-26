---
layout: post
title:  "ML Interview Questions"
date:   2025-04-01 01:00:00
categories: "interview-question"
asset_path: /assets/images/
tags: []
---

# 1. Basic Interview Questions

## 1.1 Bias - Variance Tradeoff 설명

- Bias: underfitting
- Variance: Overfitting 

중간지점을 잘 찾아서 균형을 찾는게 관건.

- Bias 줄이기
  - linear model -> Polynomial, Deep Network
  - 학습 시간 늘리기, 
  - feature engineering, loss function 개선
- Variance 줄이기
  - L1, L2, Dropout 같은 regularization 추가
  - pruning, depth 줄이기
  - 더 많은 데이터 추가
  - Bagging, Boosting 같은 ensemble learning 사용 (random forest)

**수학적으로 설명**

 - Bias: 모델 평균 예측값 -  실제 f(x) 차이 : $$ \left( f(x) - E\left[ \hat{f}(x) \right] \right)^2 $$ 
 - Variance: 모델 예측값 - 모델 예측값의 평균값  : $$ E \left[ \left(\hat{f}(x) - E\left[ \hat{f}(x) \right] \right)^2 \right] $$

Bias Variance Decomposition 저 위의 공식이 나옴. 
제곱 있는건 MSE에서  decomsition 해서 그러함. 

<img src="{{ page.asset_path }}bias-variance-math-formula.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


# 2. ML Performance 

## 2.1 ROC AUC

**ROC**

- ROC: Receiver Operating Characteristic curve 
- X축: FPR (False Positive Rate) $$ FPR = \frac{FP}{FP + TN} $$
- Y축: TPR (True Positive Rate) $$ TPR = \frac{TP}{TP + FN} $$ 
- threshold 를 움직이면서 각각을 모두 계산하는 방법
  - ex) threshold = 0.7 이면 0.7이상은 positive 이고, 이하는 negative 로 설정 

**ROC AUC**

 - ROC Curve 아래 면적 (Area Under Curve)
 - 수치
   - 0.5: 랜덤
   - 1.0: 완벽 구분
   - 0: 완전히 반대로 예측

<img src="{{ page.asset_path }}ROC-curves-and-area-under-curve-AUC.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

## 2.2 PrAUC


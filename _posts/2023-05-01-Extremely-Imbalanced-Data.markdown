---
layout: post
title:  "Extremely Imbalanced Data Training"
date:   2023-05-01 01:00:00
categories: "machine-learning"
asset_path: /assets/images/
tags: []
---

# Imbalanced Data Training with PrAUC

- [Code](https://github.com/AndersonJo/machine-learning/blob/master/121%20Imbalanced%20Model/imbalanced.ipynb)


## Result

- 10% 미만의 Imbalanced Dataset 에서는 UnderSampling 이 독약이 될 수 있다. 
- Mild 한 상황의 Imbalanced Dataset에서는 데이터 손실이 너무 크지 않아서 효과가 있을수 있지만, 극단적으로 적은 상황에서는 UnderSampling 이 작동을 안한다. 
- 이유는 단순하다. 데이터 손실 때문이다. 
- 괜찮게 쓸만한 방법이 class weight 를 조정하는 것이다. 

| Model               | Method        | Accuracy | Precision | Recall | F1 Score | PrAUC  | Description                                   |
|:--------------------|:--------------|:---------|:----------|:-------|:---------|:-------|:----------------------------------------------|
| Logistic Regression |               | 0.8963   | 0.909     | 0.0021 | 0.0042   | 0.2436 | High Accuracy and Low Recall                  |
| Logistic Regression | class weight  | 0.8963   | 0.909     | 0.0021 | 0.0042   | 0.2436 | class_weight 주던 말던 동일함                        |
| Logistic Regression | UnderSampling | 0.6158   | 0.1464    | 0.5595 | 0.2321   | 0.2227 | PrAUC 가 그냥 돌릴때보다도 적음. 웃기네                     |
| Decision Tree       |               | 0.9192   | 0.7497    | 0.3328 | 0.4631   | 0.5187 | max_depth 조절에 따라 다름                           |
| Decision Tree       | UnderSampling | 0.8301   | 0.3459    | 0.7142 | 0.4660   | 0.4847 | PrAUC 가 그냥 돌릴때보다 더 적음.                        |
| LightGBM            |               | 0.9255   | 0.8247    | 0.3595 | 0.5008   | 0.6565 | n_estimators=400 / 학습 1초도 안걸림                 |
| LightGBM            | class weight  | 0.9258   | 0.8338    | 0.3565 | 0.4995   | 0.6625 | LightGBM은 class weight 조정 하나 안하나 비슷함          |
| LightGBM            | UnderSampling | 0.89     | 0.479     | 0.6789 | 0.5617   | 0.6428 | PrAUC가 그냥 돌릴때보다도 못함                           |
| Deep Learning       |               | 0.9242   | 0.8436    | 0.3315 | 0.4760   | 0.5301 | 7.2K params / 그냥 돌림                           |
| Deep Learning       | class weight  | 0.8727   | 0.4394    | 0.8184 | 0.5718   | 0.6785 | 7.2K params / class weight adjusted 8.633 사용  |
| Deep Learning       | UnderSampling | 0.8885   | 0.4767    | 0.7508 | 0.5832   | 0.6007 | 딥러닝에서도 PrAUC 가 낮게 나옴                          |
| Transformer         | class weight  | 0.8816   | 0.4594    | 0.7908 | 0.5812   | 0.6903 | 12.8k params / class weight adjusted 8.633 사용 |
| Transformer         | UnderSampling | 0.8704   | 0.4324    | 0.7938 | 0.5598   | 0.6655 | Accuracy, PrAUC 둘다 잘 안나옴                      |


**Logistic Regression**<br>

 - Recall 에서 positive label 을 하나도 못 맞췄다.
 - class weight 주나마나 동일함

<img src="{{ page.asset_path }}imbalanced-logistic-regression.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


**Decision Tree**<br>

 - 역시 Tree 가 하나라서 댕청하다

<img src="{{ page.asset_path }}imbalanced-decision-tree.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


**LightGBM**<br>

- 학습하는데. 1초도 걸리지 않는다. 
- 그러면서도 Deep Learning과 거의 유사한 퀄리티를 보여준다

<img src="{{ page.asset_path }}imbalanced-lightgbm.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


**Deep Learning**<br>

<img src="{{ page.asset_path }}imbalanced-deeplearning.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


**Transformer**<br>

 - 어차피 sequence 데이터가 아니기 때문에 (text처럼 오른쪽으로 읽어야하는 순차성), positional encoding 은 삭제를 했다.

<img src="{{ page.asset_path }}imbalanced-transformer.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">



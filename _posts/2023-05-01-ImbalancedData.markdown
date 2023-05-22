---
layout: post
title:  "Imbalanced Data Training with PrAUC"
date:   2023-05-01 01:00:00
categories: "machine-learning"
asset_path: /assets/images/
tags: []
---

# Imbalanced Data Training with PrAUC

- [Code](https://github.com/AndersonJo/machine-learning/blob/master/121%20Imbalanced%20Model/imbalanced.ipynb)


## Result 

| Model                        | Accuracy | Precision | Recall | F1 Score | PrAUC  | Description                                   |
|:-----------------------------|:---------|:----------|:-------|:---------|:-------|:----------------------------------------------|
| Logistic Regression          | 0.8963   | 0.909     | 0.0021 | 0.0042   | 0.2436 | class_weight 주던 말던 동일함                        |
| Decision Tree                | 0.9192   | 0.7497    | 0.3328 | 0.4631   | 0.5187 | max_depth 조절에 따라 다름                           |
| LightGBM                     | 0.9258   | 0.8338    | 0.3565 | 0.4995   | 0.6625 | n_estimators=400 / 학습 1초도 안걸림                 |
| Deep Learning                | 0.9242   | 0.8436    | 0.3315 | 0.4760   | 0.5301 | 7.2K params / 그냥 돌림                           |
| Deep Learning (class weight) | 0.8727   | 0.4394    | 0.8184 | 0.5718   | 0.6785 | 7.2K params / class weight adjusted 8.633 사용  |
| Transformer (class weight)   | 0.8816   | 0.4594    | 0.7908 | 0.5812   | 0.6903 | 12.8k params / class weight adjusted 8.633 사용 |


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



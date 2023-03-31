---
layout: post
title:  "AUC & PrAUC Optimization"
date:   2018-07-07 01:00:00
categories: "statistics"
asset_path: /assets/images/
tags: ['auc', 'roc', 'curve', 'sensitivity', 'performance', 'ggplot']
---

# Basic 101 


## Confusion Matrix 

| Name                                        | Short           | Formular             | Another Formula |
|:--------------------------------------------|:----------------|:---------------------|:----------------|
| Sensitivity, **Recall**, True Positive Rate | TPR (Recall)    | TPR = TP / (TP + FN) | 1 - FNR         |
| Specificity, True Negative Rate             | TNR             | TN / (TN + FP)       | 1 - FPR         |
| Precision, Positive Predictive Value        | PPV (Precision) | TP / (TP + FP)       | 1 - FDR         |
| Fallout, False Positive Rate                | FPR (Fallout)   | FP / (FP + TN)       | 1 - TNR         |
| False Discovery Rate                        | FDR             | FP / (FP + TP)       | 1 - PPV         |



## How to choose the threshold 

문제에 따라서 정하는게 틀려 질 수 있습니다.<br> 
무조건 TRP 그리고 FPR 을 최대치로 하는 방향으로 갖을 필요는 없습니다. (즉 비즈니스 요구 사항에 따라서 달라질 수도 있습니다.)<br>
하지만 보통의 경우는 AUC 를 가장 크게 가져가는 threshold 를 찾는게 보통입니다.<br>
예를 들어서 범죄자를 찾는다고 했을때, 더 낮은 FPR 을 설정하면서, 적절한 수준의 TRP을 사람의 의해서 정해질 수 도 있습니다.

따로 비즈니스적인 요구사항이 없다면 아래의 공식을 따르면 됩니다. 

## Youden's J static Optimal Threshold 

보통의 경우는 다음과 같은 공식을 따릅니다. <br>
[Youden's J statistic](https://en.wikipedia.org/wiki/Youden%27s_J_statistic) 에 따르면

$$ \begin{align}
J &= Sensitivity + Specificity - 1 \\
J &= Sensitivity + (1 - FPR) - 1 \\
J &= TPR - FPR
\end{align} $$

그냥 `TPR - FPR` 하면 됩니다.  

$$ \text{Maximize}(TPR - FPR) $$



## Optimal Threshold for imbalanced data in ROC Curve

Geometric Mean 을 사용하는 경우 `Sensitivity - Specificity` 를 합니다. 

$$ \text{Maximize}(sqrt(TPR - (1-FPR))) $$

누군가는 Imbalanced Data 에서 성능이 좋다고 하는데.. <br>
제 실험 결과에서는 그닥 좋지 않습니다.<br>








# Python Code

## Preparation

별개로 `plt.style.use('ggplot')` 괜찮습니다

```python
%config Completer.use_jedi = False

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

plt.style.use("ggplot")
```

## Data

```Python
x_data, y_data = make_classification(
    n_samples=10000, n_features=5, n_classes=2, weights=(0.8, 0.2), flip_y=0.05, random_state=1
)

x_data = pd.DataFrame(x_data, columns=["x1", "x2", "x3", "x4", "x5"])
y_data = pd.DataFrame(y_data.reshape(-1, 1), columns=["y"])
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=2
)

# Display Data
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test :", x_test.shape)
print("y_test :", y_test.shape)
display(y_data.value_counts())
display(x_data.join(y_data).head())
```

<img src="{{ page.asset_path }}prauc-image01.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


## Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver="lbfgs")
model.fit(x_train, y_train.values.reshape(-1))

y_prob = model.predict_proba(x_test)[:, 1]
```


## ROC AUC & Optimal Threshold

```python
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def make_mark(_idx, color, label):
    _max_threshold = thresholds[_idx]
    _acc = accuracy_score(y_test, y_prob >= _max_threshold)
    _f1 = f1_score(y_test, y_prob >= _max_threshold)
    label = label + f" | t={_max_threshold:.4f} | acc={_acc:.2f} | f1={_f1:.4f}"
    plt.plot(fpr[_idx], tpr[_idx], marker="o", markersize=10, color=color, label=label)


# ROC AUC
lr_auc = roc_auc_score(y_test, y_prob)
print("Logistic: ROC AUC=%.3f" % (lr_auc))


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# TPR - FPR Optimal Threshold
idx1 = np.argmax(tpr - fpr)
max_threshold = thresholds[idx1]


# G-Means Optimal Threshold
gmeans = np.sqrt(tpr * (1 - fpr))
idx2 = np.argmax(gmeans)
max_threshold = thresholds[idx2]


# AUC (위의 roc_auc_score 과 동일)
roc_auc = auc(fpr, tpr)

# F1 Scores
scores = [f1_score(y_test, y_prob > t) for t in thresholds]
idx3 = np.argmax(scores)


plt.subplots(1, figsize=(6, 5))
plt.plot(fpr, tpr, marker=".", label=f"Classifier (AUC={roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label=f"Baseline  (AUC=0.5)")

make_mark(idx1, "blue", f"TPR-FPR")
make_mark(idx2, "yellow", "G-Mean")
make_mark(idx3, "cyan", "F1")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve")
plt.legend()
print()
```

<img src="{{ page.asset_path }}prauc-image02.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

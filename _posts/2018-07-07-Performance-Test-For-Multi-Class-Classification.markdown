---
layout: post
title:  "Performance Test For Multi Class Classification"
date:   2018-07-07 01:00:00
categories: "statistics"
asset_path: /assets/images/
tags: ['recall', 'precision', 'f1', 'metric']
---


# Easy Way

## Classification Report


{% highlight python %}
from sklearn.metrics import classification_report

y_true = [0, 0, 1, 1, 1, 1, 1, 2, 2, 3]
y_pred = [0, 2, 1, 1, 2, 0, 0, 2, 2, 0]

print(classification_report(y_true, y_pred))
{% endhighlight %}


{% highlight text %}
              precision    recall  f1-score   support

           0       0.25      0.50      0.33         2
           1       1.00      0.40      0.57         5
           2       0.50      1.00      0.67         2
           3       0.00      0.00      0.00         1

    accuracy                           0.50        10
   macro avg       0.44      0.47      0.39        10
weighted avg       0.65      0.50      0.49        10
{% endhighlight %}

## Confusion Matrix


{% highlight python %}
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = [0, 0, 1, 1, 1, 1, 1, 2, 2, 3]
y_pred = [0, 2, 1, 1, 2, 0, 0, 2, 2, 0]

cm = confusion_matrix(y_true, y_pred)

print(cm)
sns.heatmap(cm, cmap='YlGnBu')
{% endhighlight %}

{% highlight text %}
[[1 0 1 0]
 [2 2 1 0]
 [0 0 2 0]
 [1 0 0 0]]
 {% endhighlight %}
 
 <img src="{{ page.asset_path }}performance-test-confusion.png" class="img-responsive img-rounded img-fluid">
 
# Performance Test
## Accuracy

$$ \text{Accuracy} = \frac{TP + TN}{N} $$

{% highlight python %}
from sklearn.metrics import accuracy_score

y_true = [0, 0, 1, 1, 1, 1, 1, 2, 2, 3]
y_pred = [0, 2, 1, 1, 2, 0, 0, 2, 2, 0]

acc = accuracy_score(y_true, y_pred)
acc_norm = accuracy_score(y_true, y_pred, normalize=False)

print('n:', len(y_true))
print(f'Accuracy           : {acc:.2}')
print(f'Normalized Accuracy: {acc_norm}')
{% endhighlight %}

{% highlight text %}
n: 10
Accuracy           : 0.5
Normalized Accuracy: 5
{% endhighlight %}


## Recall (Sensitivity, True Positive Rate)

$$ \text{True Positive Rate} = \frac{TP}{TP + FN} $$

 - 단점: 전부다 1로 예측하면, TP는 다 맞추고, FN은 0이 되면서, recall의 예측값은 1이 된다


**average parameter**
  - **None**  : 각각의 클래스마다의 recall값을 계산한다
  - **binary**: (default) binary classification에서 사용 
  - **micro** : 전체 클래스 데이터 관점에서의 total true positives, false negatives, false positives 를 계산
  - **macro** : 각각의 label마다의 recall의 unweighted mean을 계산한다. 따라서 label imbalance를 고려하지 않는다
  
  
{% highlight python %}
from sklearn.metrics import recall_score

y_true = [0, 0, 1, 1, 1, 1, 1, 2, 2, 3]
y_pred = [0, 2, 1, 1, 2, 0, 0, 2, 2, 0]


recalls = recall_score(y_true, y_pred, average=None)
recall_micro = recall_score(y_true, y_pred, average='micro')
recall_macro = recall_score(y_true, y_pred, average='macro')
recall_weighted = recall_score(y_true, y_pred, average='weighted')

print('Recalls          :', recalls)
print(f'Recall (micro)   : {recall_micro:.2}')
print(f'Recall (macro)   : {recall_macro:.2}')
print(f'Recall (weighted): {recall_weighted:.2}')
{% endhighlight %}



## Precision

$$ \text{Precision} = \frac{TP}{TP + FP} = \frac{TP}{\text{Predicted Yes}} $$

{% highlight python %}
from sklearn.metrics import precision_score

y_true = [0, 0, 1, 1, 1, 1, 1, 2, 2, 3]
y_pred = [0, 2, 1, 1, 2, 0, 0, 2, 2, 0]

precisions = precision_score(y_true, y_pred, average=None)
precision_micro = precision_score(y_true, y_pred, average='micro')
precision_macro = precision_score(y_true, y_pred, average='macro')
precision_weighted = precision_score(y_true, y_pred, average='weighted')

print('Precisions       :', precisions)
print(f'Precision (micro)   : {precision_micro:.2}')
print(f'Precision (macro)   : {precision_macro:.2}')
print(f'Precision (weighted): {precision_weighted:.2}')
{% endhighlight %}

{% highlight text %}
Precisions       : [0.25 1.   0.5  0.  ]
Precision (micro)   : 0.5
Precision (macro)   : 0.44
Precision (weighted): 0.65
{% endhighlight %}


## F1 Score

$$ \text{F1 Score} = 2 \cdot
\frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}} $$

{% highlight python %}
from sklearn.metrics import f1_score

y_true = [0, 0, 1, 1, 1, 1, 1, 2, 2, 3]
y_pred = [0, 2, 1, 1, 2, 0, 0, 2, 2, 0]

f1_score(y_true, y_pred, average=None)

f1 = f1_score(y_true, y_pred, average=None)
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print('F1       :', f1)
print(f'F1 (micro)   : {f1_micro:.2}')
print(f'F1 (macro)   : {f1_macro:.2}')
print(f'F1 (weighted): {f1_weighted:.2}')
{% endhighlight %}

{% highlight text %}
F1       : [0.33333333 0.57142857 0.66666667 0.        ]
F1 (micro)   : 0.5
F1 (macro)   : 0.39
F1 (weighted): 0.49
{% endhighlight %}
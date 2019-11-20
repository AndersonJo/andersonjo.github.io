---
layout: post
title:  "Text Classification with Naive Bayes"
date:   2016-04-27 01:00:00
categories: "machine-learning"
asset_path: /assets/images/
tags: ['conditional-probability', 'bayes-theorem', '20news', 'spam', 'ham']
---


# Introduction

많은 Machine learning 알고리즘들이 나왔지만, 그럼에도 아직도 많이 사용되는 알고리즘중의 하나는 Naive Bayes 알고리즘이 있습니다. <br>
본문에서는 Multinomial Naive Bayes에 관해서 알아보고 예제까지 해보면서 이론부터 구현까지 튜토리얼을 해보도록 하겠습니다.

# Simple Text Example

이메일의 내용을 갖고서 스팸인지 또는 햄(스팸이 아닌 이메일)인지 분류하는 데이터는 다음과 같은 예제를 갖고 있습니다.


| Text | Tag |
|:-----|:----|
| free message | Spam |
| send me a messsage | Ham |
| are you free tomorrow? | Ham |
| where is tesseract? | Spam |
| where are you now? | Ham |
| buy awesome tv | Spam |

예를 들어, `I cooked a salmon` 이런 문장인 경우 Naive Bayes Classifier를 사용해서 Spam인지 또는 Ham인지 확률을 알아내는 것이 목표입니다.


# Tokenizing Text

Text를 기계학습의 feature로 사용하기 위해서 **word frequencies**를 사용합니다. <br>
이 경우, 문장속에서 단어운 순서나, 문장의 구조에 대한 정보 손실이 일어나게 됩니다. <br>
너무 단순한 방법이 아닐까 생각이 될지 모르지만 데이터의 적을수록 단순화 하는 것이 좋으며, 꽤나 잘 작동을 합니다.

다를 방법으로는 딥러닝을 사용해서 각각의 단어마다 Word2Vec 또는 Glove를 사용하여 vector화 하는 방법이 있습니다.<br>
이 경우 문장의 구조, 단어의 순서에 대한 정보까지도 학습을 하게 되지만, 데이터가 부족할 경우 실제 학습시 overfitting이 매우 일어나기 쉽습니다.<br>
물론 drop out, l2 regularization, 레이어의 단순화, early termination, ensemble등으로 어느정도 해결할수 있지만 당연히 accuracy가 떨어지게 됩니다.

었쟀든 word frequency를 이용한 vectorization 방법은 단순하면서, 꽤나 유용한 feature engineering 입니다.


# Bayes Theorem

베이즈 이론에 관해서 자세한 설명은 [여기](http://incredible.ai/statistics/2014/03/01/Bayes-Theorem/) 를 클릭합니다.<br>
쉽게 이야기 해서 베이즈 이론을 사용하여 conditional probabilities를 계산할수 있는데.. 이때 reversed condition을 사용함으로서 문제를 좀 더 쉽게 풀수 있게 도와 줍니다.

Bayes' Theorem 공식은 다음과 같습니다.

$$ P(A|B) = \frac{P(A \cap B )}{P(B)} = \frac{P(A) P(B|A)}{P(B)} $$

예를 들어서 스팸 필터링 예제를 공식화 하면 다음과 같습니다.

$$ P(\text{Spam} | \text{Email}) = \frac{P(\text{Spam}) P(\text{Email} | \text{Spam})}{P(\text{Email})}  $$

$$ P(\text{Email}) $$ 는 normalization으로서 $$ P(\text{Email}) =
P(\text{Spam}) P(\text{Email} | \text{Spam}) + P(\text{Ham}) P(\text{Email} | \text{Ham}) $$ 과 같습니다만,
어떤 class의 확률이 더 높은지 비교하는 것이기 때문에 $$ P(\text{Email}) $$ 의 확률은 계산할 필요가 없습니다.
따라서 비교하는 2개의 공식은 다음과 같습니다.

$$ P(\text{Spam}) P(\text{Email} | \text{Spam}) $$

$$ VS $$

$$ P(\text{Ham}) P(\text{Email} | \text{Ham}) $$






# Naive Bayes

Bayes theorem의 문제는 주어지는 데이터의 종속적 관계 때문에 연상량이 급격하게 늘어나게 됩니다.<br>
예를 들어 이메일속의 단어들의 순서는 다른 단어가 나타날 확률을 의미할 수 있으며, 이는 각각의 단어가 다른 단어에 종속적임을 의미하게 됩니다.<br>
예를 들어서 "비아그라" 라는 단어는 처방과 관련된 단어가 나올 확률 또는 발기부진에 빠진 남자들을 위한 광고와 관련된 단어들이 나올 확률이 높을 것 입니다.

Naive Bayes는 이러한 현실적인 가정을 무시하고 모든 단어(또는 features)가 모두 <span style="color:red"> **독립적(Independent)이라고 가정**</span>을 합니다.<br>
물론 현실적으로 맞지는 않지만, 그럼에도 불구하고 이러한 가정은 계산량은 줄여주면서 잘 작동합니다.<br>

많은 통계학자들이 가정 자체가 틀렸는데 왜 이렇게 잘 작동하는지 많은 연구를 하였는데.. 그중 하나의 설명이 좀 개인적으로 와닿았습니다.<br>
만약 스팸을 정확하게 모두 걸러낸다면 신뢰구간 51% ~ 99%가 의미가 있는 것인가 입니다.<br>
즉 test결과 자체가 정확하다면, 매우 정확한 확률론적 계산을 하는 것 자체가 크게 중요하지 않다는 의미입니다.


## Formula

Naive Bayes의 공식은 다음과 같습니다.

$$ P(Y_L | x_1, ..., x_n) = \frac{1}{Z} P(Y_L) \prod^n_{i=1} P(x_i | Y_L) $$

- $$ Y_L $$ : 클래스를 타나내며 예제에서는 Spam 또는 Ham
- $$ X = (x_1, x_2, x_3, ..., x_n) $$ : features들로서 예제에서는 각각의 단어를 가르킴
- $$ \frac{1}{Z} $$ : scaling factor로서 계산된 결과값을 확률로 변형시켜줍니다.


하지만 위의 공식의 문제점은 $$ \prod^n_{i=1} P(x_i \| Y_L) $$ 입니다.<br>
연속적인 확률들을 모두 곱하게 되면, 값은 점점 작아지면서 underflow가 발생하게 되고 그냥 0값이 됩니다.<br>
Machine learning에서 이 문제는 흔하고, 항상 거의 동일한 방법으로 문제를 해결합니다.<br>
바로 log likelihood를 사용하면 됩니다.

$$ \frac{1}{Z} \log P(Y_L) \sum^n_{i=1} \log P(x_i | Y_L) $$


## Example

예를 들어서 "where is tesseract" 라는 문장이 Spam 인지 Ham인지 구분하는 공식은 다음과 같습니다.

$$ P(Spam) P(Email | Spam) = P(Spam)\ P(where | Spam) \
P(is | Spam) \ P(tesseract | Spam) $$

위의 공식은 반드시 Ham일 확률과 비교해야 됩니다. 둘 중에서 확률이 높은 것으로 Spam 인지 Ham인지 결정이 됩니다.

$$ P(Ham) P(Email | Ham) = P(Ham)\ P(where | Ham) \
P(is | Ham) \ P(tesseract | Ham) $$

## Laplace Smoothing

또다른 문제가 있습니다. 예를 들어서 Ham으로 구분되는 텍스트 중에 `tesseract` 라는 단어가 없을때 입니다.<br>
이 경우 확률은 0이 되고 예를 들어서 다음과 같이 공식이 만들어 질 수 있습니다.

$$ P(Ham) P(Email | Ham) = P(Ham)\ P(where | Ham) \
P(is | Ham) \ \times 0 $$

즉 0이 곱해지기 때문에 결과값은 다른 단어들의 확률과 상관없이 0이 되게 됩니다.<br>
이를 방지하기 위해서 모든 word count에 1을 더합니다. <br>
그리고 분모에는 해당 클래스에 속하는 모든 단어들의 갯수 + 해당 단어가 나온 갯수를 divisor로 사용합니다. <br>
따라서 결과값은 항상 확률로 나오게 됩니다.

Laplace Smoothing 공식은 다음과 같습니다.

$$ P(w | c) = \frac{\text{count}(w, c) + 1}{\text{count(c)} + V + 1} $$

- count(w, c) : 해당 클래스 안에서 나온 단어의 횟수. (중복도 포함)
- count(c)    : 해당 클래스의 단어의 총 횟수 (중복도 포함. 즉 apple이 여러 문장에서 12번 나오면 12번으로 침)
- V : 전체 유니크 단어의 갯수 (중복 X)

예를 들어서 `where` 이라는 단어의 laplace smoothing을 적용한 경과는 다음과 같습니다.

$$ P( where | Spam ) = \frac{1 + 1}{12 + 20 + 1} $$



# Spam Filtering with Python

* Spam Filtering with Naive Bayes 전체 코드는 [여기](https://github.com/AndersonJo/text-classification-tutorial/blob/master/spam-filtering-with-naive-bayes.ipynb) 를 클릭합니다.
* 20 News Classification 전체 코드는 [여기](https://github.com/AndersonJo/text-classification-tutorial/blob/master/20-news-classification.ipynb) 를 클릭합니다.

## Tokenizing

{% highlight python %}

import string
from nltk.corpus import stopwords


def process_text(text):

    # Remove Punctuations
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    # Remove stopwords
    cleaned_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return cleaned_words

data['text'].apply(process_text).head()
{% endhighlight %}


## Model

{% highlight python %}
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


pipeline = Pipeline([
    ('vectorization', CountVectorizer(analyzer=process_text)),  # Convert strings to frequency vectors
    ('tfidf', TfidfTransformer()),  # Convert vectors to weighted TF-IDF scores
    ('classifier', MultinomialNB())
])
{% endhighlight %}

## Training

{% highlight python %}
train_x, test_x, train_y, test_y = train_test_split(data['text'], data['class'], test_size=0.2)
pipeline.fit(train_x, train_y)
{% endhighlight %}

## Predict

{% highlight python %}
from sklearn.metrics import classification_report,confusion_matrix


pred_y = pipeline.predict(test_x)

print(classification_report(test_y, pred_y))
sns.heatmap(confusion_matrix(test_y, pred_y), fmt='.1f', annot=True)
{% endhighlight %}


{% highlight bash %}
             precision    recall  f1-score   support

        ham       0.96      1.00      0.98       963
       spam       1.00      0.75      0.86       152

avg / total       0.97      0.97      0.96      1115
{% endhighlight %}

<img src="{{ page.asset_path }}text-classification-spam.png" class="img-responsive img-rounded img-fluid">
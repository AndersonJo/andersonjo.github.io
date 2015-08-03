---
layout: post
title:  "Naive Bayes"
date:   2015-08-04 02:00:00
categories: "machine-learning"
asset_path: /assets/posts/Naive-Bayes/

---
Bayes 공식이나 이론은 이미 [Monty Hall][bayes] 문제를 풀면서 설명을 했습니다.

오늘은 이 Bayes 공식을 이용한.. 정말 간단하지만 왠만한 복잡도 높은 다른 classification methods 보다 강력한 퍼포먼스(정확도)를
보여주는 Naive Bayes를 예제와 함께 설명하겠습니다.

일단 Naive Bayes의 공식은 다음과 같습니다.

<img src="{{page.asset_path}}naive-bayes-formula.gif" class="img-responsive img-rounded">

1. C<sub>L</sub> 은 클래스 또는 분류를 나타냅니다.
2. F 는 features
3. 1/Z 는 scaling factor



<img src="{{page.asset_path}}what-the-fuck.jpg" class="img-responsive img-rounded">

대체 뭔 소리여 ㅋㅋ 그냥 예제하나 풀면 다 이해됩니다.


# SMS Spam Classification

1. [데이터 다운로드][data]

> 데이터는 [Unicamp][unicamp] 에서 가져왔습니다.



[bayes]: /machine-learning/2015/07/29/Monty-Hall-Problem/
[data]: {{page.asset_path}}SMSSpamCollection.txt
[unicamp]: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/


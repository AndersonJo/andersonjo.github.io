---
layout: post 
title:  "Basic Statistics 101 for ML Engineers"
date:   2021-10-29 01:00:00 
categories: "machine-learning"
asset_path: /assets/images/ 
tags: []
---


<header>
    <img src="{{ page.asset_path }}coffee_keyboard.jpeg" class="center img-responsive img-rounded img-fluid">
</header>


# 1. Introduction

지난번 글에서 [머신러닝 엔지니어가 알아야할 기초 101](/engineering/2021/09/20/Basic-Engineering) 공유 했는데요.<br> 
이번 글에서는 머신러닝 엔지니어가 알고 있어야 하는 통계 부분에 대해서 공유 하겠습니다.<br>
오늘은 머신러닝 엔지니어가 반드시 알고 있어야 하는 기초 통계를 작성합니다. 


해당글은 작성중입니다.


# 2. Probability

## 2.1 쉬운 문제

**문제 01. 두개의 주사위를 굴릴때, 합이 7일 경우의 확률은?** 

> 일단 6 x 6 = 36 가지의 경우의 수가 있으며, 7이 나오는 경우는..<br>
> (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1) .. 총 6가지 존재 합니다.<br>
> 따라서 7이 나올 확률은 6/36 의 확률입니다. 


**문제 02. 세일즈맨이 9군데의 상점중 5군데를 방문한다고 할때, 몇가지 경우수가 있습니까?** 

> Permutation을 물어보는 것이고 ABCDE 나 EDCBA 는 서로 다른 것으로 취급합니다. <br>
> $$ {}^{9} \mathrm{ P }_5 = 9! / (9-5)! = 9 \times 8 \times 7 \times 6 \times 5 = 15120 $$ <br> 


**문제 03. 4개의 blue 구슬, 6개의 red 구슬 중에서 2개의 blue 구슬과, 1개의 red 구슬을 꺼낼 확률은?**

> 2개의 파란구슬, 1개의 빨간구슬 꺼낼 경우의 수는 36가지가 있으며 <br>
> $$ {}^{4} \mathrm{ C }_2 \times {}^{6} \mathrm{ C }_1 = (4 \times 3)/2 \times 6 = 36 $$ <br>
> 
> 전체 10개중에 3개를 고를 경우의 수는 120가지가 존재 합니다. <br>
> $$ {}^{10} \mathrm{ C }_3 = (10 \times 9 \times 8) / (3 \times 2 \times 1) = 120 $$
> 
> 따라서 확률은 30%가 됩니다. <br>
> $$ 36 / 120 = 0.3 $$


**문제 04** 52장의 카드속에서, 2개를 꺼냈는데, 모두 kings 일 확률은? 

> Combination 공식 사용하고, 왕 4장중에서 2장 꺼내는 확률을 구하면 됨<br>
> $$ n(S) = {}^{52} \mathrm{ C }_2 = n! / (n-r)!r! = (52 \times 51) / (2 \times 1) = 1326 $$  <br>
> 즉 52장 중에서 2장을 꺼낼 경우의 수는 1326가지 입니다. <br>
> 
> 이후 4개의 왕중에서 2개의 선택할 확률은 <br>
> $$ n(King) = {}^4 \mathrm{ C }_2 = (4 \times 3) / (2 \times 1) = 6 $$
> 
> 결론적으로 <br>
> $$ P = n(King) / n(S) = 6 / 1326 = 1/221 $$


## 2.2 Bernoulli 그리고 Binomial distribution 의 차이

Bernoulli distribution의 경우는 n=1 있는 경우이고, <br>
Binomial distribution의 경우는 독립동일분포를 갖는 (Independent and Identically Distributed) Bernoulli variables의 합입니다. 

예륻 들어서 **단 한번만 동전을 던져서** 동전의 앞면이 나올 확률을 구하는 것은 베르누이 확률 분포를 의미하며, <br>
**5번을 던져서** 정확하게 3번의 앞면이 나올 확률을 구하는 것은 binomial distribution을 구하는 것입니다. 

참고로 5번을 던져서 모두 앞면이 나올 확률은 p^5 이며 0.5^5 = 0.03125 이며,<br>
정확하게 3번이 나올 확률을 구하는 것은 아래 공식과 같습니다.

$$ \begin{align}
P(r) &= \begin{pmatrix} 
n \\
r \\
\end{pmatrix}
\frac{n!}{(n-r)!r!}\ p^{r} q^{n-r} \\ 
&= combination * 성공확률^{성공횟수} * 실패확률^{실패횟수} \\ 
&= \frac{5!}{(5-3)! 3!}(0.5)^{3}(0.5)^{5-3} \\ 
&= 0.3125
\end{align} $$

{% highlight python %}
> from scipy.stats import binom 
> binom.pmf(3, 5, 0.5)
0.3125
{% endhighlight %}























# 3. Statistics 

## 3.1 Statistical Significance 

통계적 유의성 (statistical significance)을 찾기 위해서 보통 가설 검증(hypothesis)를 사용하며, 
확률적으로 "우연히" 발생하지 않은 것으로 생각되는 의미있는 결과를 말합니다. <br>
보통 p-value 가 5% 이하일때 통계적 유의성이 존재한다고 합니다.


## 3.2 Bias and Variance 

아래 링크를 참고 합니다. <br>
[Ensemble / Bias and Variance Tradeoff / Bagging VS Boosting](/machine-learning/2015/10/25/Bias-Variance-Tradeoff-Ensemble/)



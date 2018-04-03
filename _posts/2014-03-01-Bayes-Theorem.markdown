---
layout: post
title:  "Bayes Theorem"
date:   2014-03-01 01:00:00
categories: "statistics"
asset_path: /assets/images/
tags: ['Conditional Probability', 'Posterior', 'Prior', 'probability']

---

#  Basic Probability

## Events (Union, Intersection, and Disjoint Events)

### Union Events

예를 들면 주사위를 던졌을때 4 또는 6이 나올 확률로 정의를 내릴 수 있다.  <br>
Event C는 2 이벤트의 합집합(union)이다.

* Event A = 4가 나올 확률
* Event B = 6이 나올 확률

$$ P(C) = P(A \cup B) $$

<img src="{{ page.asset_path }}bayes_union.png" class="img-responsive img-rounded img-fluid">


### Intersection Events

주사위를 2번 던졌을때 2 그리고 3이 나올 확률로 생각하면 된다.  <br>
곱 연산을 한다.

* Event A = 2가 나올 확률
* Event B = 3이 나올 확률

$$ P(C) = P(A \cap B) $$

<img src="{{ page.asset_path }}bayes_intersection.png" class="img-responsive img-rounded img-fluid">




## Independent, Dependent Events

두개의 이벤트 A 그리고 B 이벤트가 있다고 가정할때, 서로 영향을 미치지 않을때 independent events라고 합니다.<br>
예를 들어서 다음과 같은 이벤트들을 independent events라고 합니다.

1. 동전의 앞면 나오기 **그리고** 주사의 5 얻기
2. 통안에서 빨간 구글 얻기 **그리고** 동전의 앞면 나오기

### Probability of Independent Events

$$ P(A \cap B) = P(A) * P(B)  $$

* Event A : 4개의 빨간 구술과, 3개의 파란 구슬을 갖고 있는 통안에서 빨간구슬을 집기
* Event B : 동전의 앞면 나오기

위의 2가지 이벤트 모두 성립(빨간 구슬 얻고, 동전의 앞면 나오기)할때 게임에서 이긴다면 이길 확률은?

* $$ P(A) = \frac{4}{7} $$ .
* $$ P(B) = \frac{1}{2} $$ .

$$ \begin{align}
P(A \cap B) &= P(A) * P(B) \\
&= \frac{4}{7} * \frac{1}{2} \\
&= \frac{2}{7}
\end{align} $$

### Probability of Dependent Events

통안에서 2번 구슬을 꺼낼때를 가정할때, 첫번째 집은 구슬에 따라서 2번째 집는 구슬의 확률이 달라지게 됩니다.<br>
가령 빨간구슬을 집을 확률이 4/7이고, 실제로 빨간 구슬을 얻었으면 두번째 다시 빨간 구슬을 얻을 확률은 3/6이 됩니다.<br>
하지만 첫번째 파란 구슬을 집었다면, 두번째 빨간 구슬을 집을 확률은 4/6 이 됩니다.<br>
즉 두번째 이벤트는 첫번째 구슬을 꺼내는 이벤트에 의해서 영향을 받게 됩니다.



### Mutually Exclusive and Collectively Exhaustive Events


**Mutually Exclusive Events**란 두개의 이벤트가 동시에 일어날수 없는 것입니다.<br>
예를 들어서 동전을 던졌을때 `앞면` 그리고 `뒷면` 이 동시에 나올 수 없습니다.

**Collectively Exhaustive**가 되기 위해서는 모든 가능성을 포함해야 합니다. <br>
주사위를 예로 들면 [1,2,3,4,5,6] 이 exhuastive collection이 됩니다. 모든 가능성을 모두 포함하고 있죠.

예를 들어서, P(even) = [2, 4, 6] 그리고 P(prime) = [2, 3, 5] 일때 1이 빠졌기 때문에 exhuastive가 아닙니다.




# Conditional Probability

조건부 확률은 어떤 이벤트 B가 주어졌을때 A의 확률을 알아냅니다.

$$ P(A|B) = \frac{P(A\ \text{and}\ B)}{P(B)} = \frac{P(A \cap B)}{P(B)} = \frac{P(B \cap A)}{P(B)} $$



### Example 1

Short Time Warm-up을 갖었을때 Deb이 이길 확률은?

| Warm-up Time | Deb Wins | Bob Wins | Total |
|:-------------|:---------|:---------|:------|
| Short | 4 | 6 | 10 |
| Long  | 16 | 24 | 40 |
| Total | 20 | 30 | 50 |

$$ P(Deb | Short) = \frac{P(Deb \cap Short)}{P(Short)} = \frac{\frac{4}{50}}{ \frac{10}{50}} = \frac{2}{5} = 0.4 $$





# Reversing the Condition

Anderson은 아침에 옥수수를 먹는 것을 좋아하고, 점심에는 피자 먹는 것을 좋아합니다.<br>
아침에 옥수수를 먹을 확률 A는 0.6이고, 점심에 피자를 먹을 확률 B는 0.5 입니다.<br>
점심에 피자를 반드시 먹는다면, 아침에 옥수수를 먹을 확률은 0.7 입니다.

* P(옥수수) = 0.6
* P(피자) = 0.5
* P(옥수수 \| 피자) = 0.7

이 부분에서 **P(옥수수) != P(옥수수 \| 피자)** 이므로 아침에 옥수수먹는 일과 점심에 피자를 먹는 일은 **dependent** 하다고 볼 수 있습니다.<br>
만약 P(A) = P(A|B) 라면 서로의 이벤트들은 independent하다고 볼 수 있습니다.


이제 이러한 상황을 알 고 있고, **반대로 아침에 옥수수를 먹었고, 점심에 피자를 먹을 확률  P(피자 \| 옥수수)** 을 알고자 한다면 Bayes Theorem이 필요로 하게 됩니다.






# Bayes Theorem

Bayes Theorem은 원래 역확률(inverse problem)을 해결하기 위한 방법입니다. <br>
즉 조건부 확률 $$ P(B|A) $$ 를 **알고 있을때**, 서로 반대인 $$ P(A|B) $$ 의 확률을 알고자 할때 사용하는 방법입니다.<br>


$$ P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{P(A)P(B|A)}{P(B)} $$

* P(A \| B) : Conditional Probability 입니다. B가 주어졌을때 A가 일어날 확률
* P(B \| A) : 이것또한 Conditional Probability 입니다. A가 주어졌을때 B가 일어날 확률

중요하게 볼 부분은 Bayes's formula 는 두개의 서로다른 conditional probabilities (P(A | B) 그리고 P(B | A)) 를 서로 연결 시키며, 궁극적으로 조건이 되는 부분을 서로 뒤집어 버립니다.
결론적으로 Bayes는 conditional probability를 이용하며, 공식이 동일하며, 이때 **역확률 (inverse probability 또는 Posterior probability)**을 알고자 할때 사용한다는 것이 핵심 포인트입니다.

위의 공식에서 조금 더 일반화를 하면 다음과 같이 쓸 수 있습니다.

$$ \begin{align}
P(A_k | B) &= \frac{P(A_k \cap B)}{P(A_1 \cap B) + P(A_2 \cap B) + ... + P(A_n \cap B)}    \\
&= \frac{P(A_k) P(B|A_k)}{ \sum^n_{i=1} P(A_i) P(B|A_i) }
\end{align} $$






### Example 1

|      | Sedan | SUV | Total |
|:-----|:---------|:---------|:------|
| New  | 24 | 15 | 39 |
| Used | 9  | 12 | 21 |
| Total | 33 | 27 | 60 |


세단이라는 조건하에, 랜덤으로 선택했을때 New 가 선택될 확률?

$$ P(New|Sedan) = \frac{P(New)P(Sedan|New)}{P(Sedan)} = \frac{0.65 * 0.6154}{0.55} = 0.727 $$





### Example 2

정아는 내일 결혼을 합니다. 결혼식 장소는 사막이기 때문에 비는 거의 내리지 않으며, 평균적으로 1년에 5일정도 내립니다. <br>
불행하게도 기상캐스터는 바로 결혼식날인 내일 비가 온다고 예보를 하였습니다. <br>
통계적으로 기상캐스터가 비가 내린다고 예보한뒤 실제로 비가 내린 확률은 90%이며, 실제로 비가 내리지 않는데 비가 내린다고 잘못 예보한 경우는 10%입니다. 그렇다면 내일 결혼식날에 비가 내일 확률은 얼마나 될까요?

* P(rain) = 5/365 = 0.0136985  (정아의 결혼식에 비가 내릴 확률)
* P(not rain) =  360/365 = 0.9863014  (정아의 결혼식에 비가 내리지 않을 확률)
* P(C \| rain) = 0.9  (실제 비가 온날, 기상캐스터가 정확하게 비가 온다고 예측한 확률)
* P(C \| not rain) = 0.1  (비가 내리지 않는 날, 기상캐스터가 비가 온다고 잘못 예보한 확률)

알고싶은것은 P(rain \| C) 입니다.

$$ \begin{align}
P(\text{rain}\ |\ \text{C}) &= \frac{P(\text{rain}) P(\text{C}\ |\ \text{rain})}{ P(\text{rain}) P(\text{C}\ |\ \text{rain}) + P(\text{not rain}) P(\text{C}\ |\ \text{not rain})} \\
&= \frac{0.014 * 0.9}{0.014*0.9 + 0.986 * 0.1 } \\
&= 0.111
\end{align} $$

반직관적인 결론인데.. 기상캐스터가 내일 당장 비가 온다고 하였더라도 실제 비가 올 확률은 오직 11%에 불과합니다.
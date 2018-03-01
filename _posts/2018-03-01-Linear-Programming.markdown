---
layout: post
title:  "Linear Programming"
date:   2018-03-01 01:00:00
categories: "quant"
asset_path: /assets/images/
tags: ['google', 'gcm', 'BigQuery']
---


# Anderson 공장 예제

바로 예제를 통해서 배워보도록 하겠습니다. <br>
Anderson 공장은 로보트를 만드는 공장이며, 로봇팔 A 그리고 로봇다리 B 를 생산합니다. <br>
여기 회장 Anderson은 A 와 B를 얼마만큼 생산했을때 분기별 가장 큰 이익을 얻는지를 알고 싶어합니다.

다만 여기에는 다음과 같은 제약사항들이 있습니다. <br>

1. 두 제품 모두 CPU가 필요합니다. 공급자로 부터 분기마다 받을 수 있는 최대 물량은 10,000개 입니다.
2. 두 제품 모두 SSD가 필요합니다. A에는 SSD 1개 그리고 B에는 SSD 2개가 필요하며 확보가능 물량은 15,000개 입니다.
3. 제품생산에는 시간이 소요됩니다. A는 4분, B는 3분이 걸리며, 각 분기별 가용시간은 25,000분 입니다.

제품 A는 750달러 그리고 제품 B는 1000달러에 판매가 되고 있습니다. <br>
자 다시 질문을 하자면.. "다음 분기에 Anderson 공장은 각각의 제품을 몇개씩 생산을 해야지 최대의 이익을 내는지 알고자 합니다."



## Modeling

어떤 문제를 linear programming으로 모델링 하기 위해서는 몇가지 조건들을 만족시켜줘야 합니다.



* **Decision Variables**: 여기 예제에서는 제품 A의 갯수, 제품 B의 갯수라고 생각하면 되며, $$ x_1 $$ 그리고 $$ x_2 $$ 로 나타낼수 있습니다.
* **Objective**: 모든 linear program 에는 objective가 존재하며 minimized 되거나 maximized 됩니다. <br>objective는 반드시 linear해야 합니다. 즉.. decision variables의 상수곱의 합이 되어야 합니다. <br>예를 들어 $$ 3x_1 - 10x_2 $$ 는 linear function 입니다. 하지만 $$ x_1 x_2 $$ 는 linear이 아닙니다.
* **Constraints**: 예제에서는 CPU, SSD, 소요시간, Non-negativity 의 제약조건들이 있습니다.

최종적인 모델의 공식은 다음과 같습니다.<br>


$$  \max_{(profit)} \text{profit}  =  750 x_1 + 1000 x_2 $$

제약조건은 다음과 같습니다. (1000 단위는 -> 1로 나타냅니다.)

$$ \begin{align}
x_1 + x_2 &\le 10 & \text{CPU} \\
x_1 + 2x_2 &\le 15 & \text{SSD} \\
4x_1 + 3x_2 &\le 25 & \text{Assembly time} \\
x_1 &\ge 0 & \text{제품 A 갯수} \\
x_2 &\ge 0 & \text{제품 B 갯수} \\
\end{align} $$


## Code

* **C** : Objective function에 들어가는 coefficients로서 n x 1 매트릭스 형태입니다. <br> $$ \min_{(x_1, x_2)} 2x_1 + x_2 $$ 라면 C값을 [2, 1] 로 설정하면 됩니다. <br>만약 maximize하고자 한다면 음수로 곱해주면 됩니다.
* **G** : G는 제약조건의 coefficients 로서 m x n 매트릭스입니다. <br> $$ G*x \le h $$ 이므로 G 와 h 에 음수로 곱해주면 $$ -G*x \ge -h $$ 로 바뀌게 됩니다.
* **H** : H는 G와 연동이 되는데 $$ G*x \le h $$ 에서 우측항에 해당이 됩니다.

{% highlight python %}
import numpy as np
from cvxopt import matrix, solvers

C = matrix([-750., -1000.])
G = matrix([[1., 1., 4., -1., 0.], [1., 2., 3., 0., -1.]])
H = matrix([10., 15., 25., -0., -0.])

solver = solvers.lp(C, G, H)
d = solver['x']
a, b = np.array(np.round(d) * 1000, dtype=np.int)
a, b = a[0], b[0]

display(solver)
print('[최대 이익 생산]')
print(f'제품 A: {a}개')
print(f'제품 B: {b}개')
{% endhighlight %}

{% highlight text %}
[최대 이익 생산]
제품 A: 1000개
제품 B: 7000개
{% endhighlight %}



# 마케팅 비용 예제

Anderson 마케팅 회사는 TV광고 그리고 온라인에 마케팅 캠패인을 진행하려고 합니다.<br>
다음의 도표는 비용당 다음의 유저 segment에 광고가 노출되고 앱을 다운로드 받게 됩니다.

| 매체   | 10대 | 20대 | 30대 | 비용 |
:-------|:----|:-----|:-----|:----|
| TV    | 5명   | 1명    | 3명    | 60\$  |
| 온라인 |  2명  | 6명    | 3명    | 50\$   |
| 목표 (앱 다운로드)  | 24  | 18  | 24 | |

* **Decision Variables**: TV광고 $$ x_1 $$ 그리고 인터넷광고 $$ x_2 $$ 2개로 나뉩니다.
* **Objective**: CPA(Cost per Acquisition)기준으로 $$ 60 x_1 + 50 x_2 $$
* **Constraints**: 각각의 나이별 다운로드 목표를 만족시켜야 하며, 비용은 최소한으로 써야 합니다.

공식은 다음과 같이 될 수 있습니다.

$$ \min_{\text{acquisition}} 60 x_1 + 50 x_2 $$

제약조건은 다음과 같습니다.

$$ \begin{align}
5x_1 + 2x_2 &\ge 24 & \text{10대} \\
x_1 + 6x_2 &\ge 18 & \text{20대} \\
3x_1 + 3 x_3 &\ge 24 & \text{30대} \\
x_1 &\ge 0 & \text{TV} \\
x_2 &\ge 0 & \text{온라인} \\
\end{align} $$



{% highlight python %}
C = matrix([60., 50.])
G = matrix([[-5., -1., -3., -1., -0.], [-2., -6., -3., -0., -1.]])
H = matrix([-24., -18., -24., -0, -0])

solver = solvers.lp(C, G, H)
a, b = np.round(solver['x'], 2) * C
a, b = a[0], b[0]

print('\n조건을 만족시키는 최소한의 캠페인 집행 비용')
print(f'TV광고 비용:\t{a}$')
print(f'온라인 비용:\t{b}$')
{% endhighlight %}

{% highlight text %}
조건을 만족시키는 최소한의 캠페인 집행 비용
TV광고 비용:	160.2$
온라인 비용:	266.5$
{% endhighlight %}
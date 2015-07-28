---
layout: post
title:  "Monty Hall Problem - Bayes"
date:   2015-07-29 22:00:00
categories: "machine-learning"
asset_path: /assets/posts/Monty-Hall-Problem/
---

<img src="{{page.asset_path}}saw-play-a-game.jpg" class="img-responsive img-rounded">

깨어나보니 직소가 눈앞에 있다!! 훨~ <br>
그리고 직소는 당신과 게임을 하길 원한다.<br>
게임의 룰은 다음과 같다.

1. 3개의 문이 있다.
2. 단 하나의 문 뒤에 살아나갈수 있는 열쇠가 있다.
3. 게임의 시작은 내가 먼저 열쇠가 있을거 같은 문을 선택을 한다. (선택만하고 문은 열지 않는다)
4. 직소는 선택하지 않은 문 2개중에 열쇠가 없는 문을 연다. (즉 직소는 당연히 열쇠가 어디 있는지 알고 있다.)<br>
5. 그리고 당신에게 선택이 주어진다. 처음에 선택한 문을 열것인가? 아니면 결정을 바꾸고 다른 문을 선택할 것인가?
6. 열쇠가 있는 문을 선택하면 살고, 뒤에 아무것도 없으면 죽는거임

자.. 먼저 아래의 내용을 보시기 전에 당신의 선택은 어떻게 하시겠습니까?


# Conditional Probability

보통 2개의 이벤트(상황)이 있는데 서로 영향을 미치는(dependent) 상황일때 사용을 합니다.
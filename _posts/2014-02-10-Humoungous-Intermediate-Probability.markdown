---
layout: post
title:  "Humoungous - Intermediate Probability"
date:   2014-02-10 01:00:00
categories: "statistics"
asset_path: /assets/posts/Humoungous-Advanced-Probability/
tags: []

---

# Binomial Probability Distribution

* Binomial의 이름에서 알 수 있듯이, 2개의 결과값을 갖는다 (주로.. 성공 또는 실패)

$$ P(r) = \begin{pmatrix} 
n \\
r \\
\end{pmatrix}
\frac{n!}{(n-r)!r!}\ p^{r} q^{n-r} = \begin{equation} combination * 성공확률^{성공횟수} * 실패확률^{실패횟수} \end{equation}$$

| n | 시도 횟수 | 고정값 fixed number of trials | 
| r | 성공 횟수 | 변수 |
| p | 성공 확률 | 상수 - 즉 고정값 |
| q | 실패 확률 | 상수 - 즉 고정값 |

> Q. 동전을 6번 던져을때 정확하게 두번 heads가 나오는 확률은? <br>
> (23.4%의 확률로 정확하게 2번 heads가 나올수 있다.)

$$ P(r) = \begin{pmatrix}
6 \\
2 \\
\end{pmatrix} 
\frac{6*5}{2*1}(0.5)^{2}(0.5)^{4}
= 15 * 0.25 * 0.0625 = 0.234375
$$













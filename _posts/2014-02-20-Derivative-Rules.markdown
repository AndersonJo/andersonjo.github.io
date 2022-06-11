---
layout: post
title:  "Derivative Rules"
date:   2014-02-10 01:00:00
categories: "mathematics"
asset_path: /assets/images/
tags: ['linear', 'gradient', 'loss']

---

# 1. Derivatives Rules

## 1.1 Rules

derivative of x 에 대한 테이블

| Rule                       | Function        | Derivative         |
|:---------------------------|:----------------|:-------------------| 
| Constant                   | c               | 0                  |
| Line                       | x               | 1                  |
|                            | ax              | a                  |
| Square                     | $$ x^2 $$       | $$ 2x $$           |
| Square Root                | $$ \sqrt{x} $$  | $$ ½ x^{-½} $$     |
| Exponential                | $$ e^x $$       | $$ e^x $$          |
|                            | $$ a^x $$       | $$ \ln(a) a^x $$   |
| Logarithms                 | $$ \ln(x) $$    | 1/x                |
|                            | $$ \log_a(x) $$ | 1/ (x ln(a))       |
| Trigonometry (x is radian) | sin(X)          | cos(x)             |
|                            | cos(x)          | -sin(x)            |
|                            | tan(x)          | $$ sec^2(x) $$     |
| Inverse Trigonometry       | $$ sin^-1(x) $$ | 1/ \sqrt{(1-x^2)}  |
|                            | $$ cos^-1(x) $$ | -1/\sqrt{(1-x^2)}  |


Function Rules!

| Rule                      | Function        | Derivative                  |
|:--------------------------|:----------------|:----------------------------|
| Mutiplication by constant | cf              | cf`                         |
| Power Rule                | $$ x^n $$       | $$ nx^{n-1} $$              |
| Sum Rule                  | f + g           | f' + g'                     |
| Difference Rule           | f - g           | f' - g'                     |
| Product Rule              | fg              | fg' + f'g                   |
| Quotient Rule             | f/g             | $$ \frac{f'g - g'f}{g^2} $$ |
| Reciprocal Rule           | 1/f             | $$ \frac{-f'}{f^2} $$       |
| Chain Rule                | $$ f \cdot g $$ | $$ (f \cdot g) g' $$        |
| Chain Rule with '         | f(g(x))         | f'(g(x)) g'(x)              |

## Examples

### $$ \frac{d}{dx} x^3 $$ 에 대한 값은?

derivative of $$ x^3 $$ 은 power rule 을 적용합니다.

$$ \begin{aligned} \frac{d}{dx} x^n &= nx^{n-1} \\
\frac{d}{dx} x^3 &= 3x^{3-1} = 3x^2
\end{aligned} $$

### x^2 + x^4 에 대한 값은?

derivative of f + g = f' + g' 인 sum rule 을 사용.

$$ \begin{aligned}
\frac{d}{dx} \left[ x^2 + x^4 \right] = 2x + 4x^3
\end{aligned} $$

### Chain Rule 예제

what is the derivative of $$ f(x) = (3x + 1)^5 $$ ?

기본적으로 chain rule 은 바깥쪽에서 미분한번 하고, 안쪽에서 다시 미분하고.. 서로 곱하면 됨. <br>
여기서 바깥쪽은 5제곱한 것이고, 안쪽은 3x + 1 임

$$ f'(x) = 5(3x+1)^4 (3x +1)' = 5(3x+1)^4 \times 3 $$
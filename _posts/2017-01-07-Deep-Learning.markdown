---
layout: post
title:  "Deep Learning - Backpropagation"
date:   2017-01-07 01:00:00
categories: "artificial-intelligence"
asset_path: /assets/posts2/TensorFlow/
tags: ['Artificial-Intelligence']

---

<header>
    <img src="{{ page.asset_path }}neural-network.png" class="img-responsive img-rounded img-fluid">
    <div style="text-align:right;"> 
    <small>Deep Learning 에 대해서 알아보자
    </small>
    </div>
</header>


# Single Neural Network (Perceptron)

## Input

먼저 input의 weighted sum을 구합니다. <br>
공식에 bias를 따로 $$ b $$로 잡았지만, 보통 weight의 첫번째 element는 bias로 사용합니다.


$$ z = h(x; \theta, b) = \left[ \sum^K_{i=1} w_i x_i \right] + b = w^T x + b $$


**Derivative of the Weights**

$$ \frac{\partial}{\partial w} \left[ w^T x + b \right] = x$$


**Derivative of the Bias**

$$ \frac{\partial}{\partial b} \left[ w^T x + b \right] = 1$$


## Activation Function

$$ \phi $$ 함수는 activation fuction을 나타내며 예제를 위해서 sigmoid function (or logistic function)을 사용하겠습니다.

$$ \phi(z; w) = \frac{1}{1 + e^{-z}} $$

**Derivative of the sigmoid function**은 다음과 같습니다.

$$
\begin{align}
\dfrac{d}{dz} \phi(z) &= \dfrac{d}{dz} \left[ \dfrac{1}{1 + e^{-x}} \right]  \\
&= \dfrac{d}{dz} \left( 1 + \mathrm{e}^{-z} \right)^{-1}  \\
&= -(1 + e^{-z})^{-2}(-e^{-z}) \\
&= \dfrac{e^{-x}}{\left(1 + e^{-z}\right)^2} \\
&= \dfrac{1}{1 + e^{-z}\ } \cdot \dfrac{e^{-z}}{1 + e^{-x}} \\
&= \dfrac{1}{1 + e^{-z}\ } \cdot \dfrac{(1 + e^{-z}) - 1}{1 + e^{-z}}  \\
&= \dfrac{1}{1 + e^{-z}\ } \cdot \left( 1 - \dfrac{1}{1 + e^{-z}} \right) \\
&= \phi(z) \cdot (1 - \phi(z))
\end{align}
$$

* $$ -(1 + e^{-z})^{-2}(-e^{-z}) $$ :  Chain Rule을 적용
* $$ \frac{d}{dx} e^{-z} = -e^{-z} $$  이며  $$ \frac{d}{dx} e^{z} = e^{z} $$


## Cost Function 

먼저 예제로서 **Object function** $$ J(w) $$ 를 정의합니다.<br>
이때 $$ \phi(z^{(i)}) $$ 는 activation function 입니다.

$$ \begin{align} 
J(w) &= \frac{1}{N} \sum_i \left( y^{(i)} - \phi(z^{(i)}) \right)^2 \\
\end{align} $$









# Stochastic Gradient Descent

## Calculate Gradient with regard to weights

먼저 Feedforward 순서도를 그리면 다음과 같습니다.

$$ J(\phi(h(x))) $$

즉.. J는 mean squared error function, $$ \phi $$는 sigmoid function, 그리고 h 는 $$ w^T x + b $$ 입니다.<br>
이런식으로 함수 안에 다른 함수가 사용되는 부분은 chain rule로 derivation을 할수 있습니다.

Optimization 문제는 objective function을 minimize 또는 maximize하는데 있습니다. <br>
SSE를 사용시 minimize해야 하며, learning은  stochastic gradient descent를 통해서 처리를 하게 됩니다.

$$ \frac{\partial J}{\partial w_i} =
\frac{\partial J}{\partial \hat{y}} \cdot
\frac{\partial \hat{y}}{\partial z } \cdot
\frac{\partial z}{\partial w_i }
$$

즉 다음과 같다고 할 수 있습니다. <br>

> (예제로서 activation function은 sigmoid 사용, loss는 mean squared error 사용)<br>
> Stochastic gradient를 사용함으로 $$ \sum $$ 심볼은 제외 될 수 있습니다.

$$ \begin{align}
\frac{\partial J}{\partial w_i} &=
\frac{\partial }{\partial \hat{y}} \left[ \frac{1}{N} \sum_{i=1} \left( y^{(i)} - \hat{y}^{(i)}  \right)^2 \right] \cdot
\frac{\partial}{\partial z} \left[ \frac{1}{1+e^{-z}} \right] \odot
\frac{\partial}{\partial w_i} \left[ w_i^T x^{(i)} + b \right] & [1] \\
&= -\frac{2}{N} \left[ \sum_{i=1} \left( y^{(i)} - \hat{y}^{(i)} \right) \right] \odot
 \left[ \hat{y}^{(i)} \cdot (1-\hat{y}^{(i)}) \right] \cdot x^{(i)} & [2]
\end{align} $$

* **[2]** 에서 derivative of the sigmoid function은 $$ \phi(z) \cdot (1-\phi(z)) $$ 입니다.<br> 즉 $$ \phi(z) $$는 $$ \hat{y}^{(i)} $$으로 변경될 수 있습니다.
* **[2]** 에서 $$ \odot $$ 은 element wise multiplication이며 이는 action function이 element wise function 이기 때문에 동일하게 backpropagation에서도 element wise multiplication을 해주는 것입니다.




## Calculate Gradient with regard to bias 

다음과 같은 Chain Rule이 그려지게 됩니다.

$$ \frac{\partial J}{\partial b_i} = 
\frac{\partial J}{\partial \hat{y}} \cdot 
\frac{\partial \hat{y}}{\partial z } \cdot
\frac{\partial z}{\partial b_i } 
$$

풀어쓰면 다음과 같은 식이 만들어지게 됩니다.<br>

> (위와 마찬가지로 activation function은 sigmoid 사용, loss는 mean squared error 사용)<br>
> Stochastic gradient를 사용함으로 $$ \sum $$ 심볼은 제외 될 수 있습니다.

$$ \begin{align} 
\frac{\partial J}{\partial b_i}  &=
\frac{\partial }{\partial \hat{y}} \left[ \frac{1}{N} \sum_{i=1} \left( y^{(i)} - \hat{y}^{(i)}  \right)^2 \right] \odot
\frac{\partial}{\partial z} \left[ \frac{1}{1+e^{-z}} \right] \cdot
\frac{\partial}{\partial b} \left[ w^T x + b \right] \\
&= -\frac{2}{N} \left[ \sum_{i=1} \left( y^{(i)} - \hat{y}^{(i)} \right) \right] \odot
 \left[ \hat{y}^{(i)} \cdot (1-\hat{y}^{(i)}) \right] \cdot 1
\end{align} $$

* **[2]** 에서 derivative of the sigmoid function은 $$ \phi(z) \cdot (1-\phi(z)) $$ 입니다.<br> 즉 $$ \phi(z) $$는 $$ \hat{y}^{(i)} $$으로 변경될 수 있습니다.



## Update Weights

$$ \eta $$ 는 learning rate 입니다.<br>
$$ \nabla J $$ 는 위에서 나온 공식 $$ \frac{\partial J}{\partial w_i} $$ 를 가르킵니다.

$$ \begin{align}
\Delta w &= - \eta \nabla J(w)  \\
w &= w + \Delta w
\end{align}$$


# Backpropagation Algorithm

* $$ \theta $$ 는 neural network안의 모든 weights를 말합니다.
* $$ \theta^{l}_{i, j} $$ 는 l번째 weight를 가르킵니다.
* layers의 인덱스는 1 (input), 2 (hidden), ... , L (output)을 가르킵니다.


### [1] Feedforward
Feedfoward Pass를 $$ h^{(1)} $$, $$ h^{(2)} $$, $$ h^{(3)} $$, ...., $$ h^{(L)} $$ 에 대해서 계산을 합니다.

$$ \begin{align}
h^{(1)} &= x \\
h^{(2)} &= \phi \left( \left( \theta^{(1)} \right)^T h^{(1)} + b^{(1)} \right)\\
 ... \\
h^{(L-1)} &= \phi \left(  \left( \theta^{(L-2)} \right)^T h^{(L-2)} + b^{(L-2)} \right) \\
h(x) = h^{(L)} &= \phi \left( \left( \theta^{(L-1)} \right)^T h^{(L-1)} + b^{(L-1)} \right)
\end{align} $$





### [2] output layer 에서의 계산
마지막 output에서는 다음과 같은 계산을 해줍니다.

$$ \begin{align}
\frac{\partial J}{\partial \theta^{(L)}} &=
\frac{\partial J}{\partial h^{(L)}} \cdot 
\frac{\partial h^{(L)}}{\partial h^{(L-1)} } \cdot
\frac{\partial h^{(L-1)}}{\partial \theta^{(L)} } 
\end{align} $$

$$ \begin{align}
\frac{\partial J}{\partial \theta^{(L)}} = \delta^{(L)} &=  
\frac{\partial}{\partial h^{(L)}} \left[ \frac{1}{N}  \sum_{i=1} \left( h^{(L)} - y^{(i)} \right)^2 \right] \cdot
\frac{\partial}{\partial h^{(L-1)}} \left[ \frac{1}{1-e^{-h^{(L-1)}}} \right] \cdot
\frac{\partial}{\partial \theta^{(L-1)}} \left[ \left( \theta^{(L-1)} \right)^T h^{(L-1)} + b^{(L-1)} \right] & [1] \\
&= \frac{2}{N} \left[ \sum \left( h^{(L)} - y^{(i)} \right) \right] \odot \phi^{\prime} \left( (\theta^{(L-1)})^T h^{(L-1)} + b^{(L-1)} \right) & [2] \\
&= \frac{2}{N} \left[ \sum \left( h^{(L)} - y^{(i)} \right) \right] \odot h^{(L)} (1- h^{(L)}) & [3]
\end{align} $$

* **[1]** $$ \frac{1}{N}\sum $$ 부분에서 N을 2값으로 변경하여 ($$ \frac{1}{2} \sum $$) 계산 효율을 높일수 있습니다.
* **[2]** 전형적인 output derivation공식입니다.
* **[3]** $$ \phi $$ 함수에 derivative of the sigmoid function $$ \phi(z)(1-\phi(z)) $$를 적용했을때 입니다.



### [3] L-1 부터의 계산

$$ l =  L -1, L -2, L -3 $$, ... 부터 다음과 같은 계산을 합니다.<br>

$$ \begin{align}
\delta^{(l)} &= \left[ \left( \theta^{(l)} \right)^T \delta^{(l+1)} \right] \odot \phi^{\prime} \left( \left( \theta^{(l-1)} \right)^T h^{(l-1)} + b^{(l-1)} \right) & [1] \\
&= \left[ \left( \theta^{(l)} \right)^T \delta^{(l+1)} \right] \odot h^{(l)}(1-h^{(l)}) & [2]
\end{align} $$

* **[1]**에서 $$\left( \theta^{(l)} \right)^T \delta^{(l+1)}  $$ 는 feedword 에서 각 layer의 output을 받아서 다른 layer에서 연산하던 과정을 다시역으로 계산하면서 나타난 부분입니다.
* **[2]**에서 $$ \phi^{\prime} $$  은 sigmoid로 대체를 하였습니다.



### [4] Update

$$ \begin{align} 
\Delta \theta^{(l)} &= \delta^{(l+1)} \left( h^{(l)} \right)^T \\
\Delta b^{(l)} &=  \delta^{(l+1)}
\end{align} $$

# Code in Numpy

[https://github.com/AndersonJo/deep-layer](https://github.com/AndersonJo/deep-layer) 에서 전체 코드를 볼 수 있습니다.<br>
update부분에서 momentum을 적용했습니다.

### Backpropagation Code

{% highlight python %}
def backpropagation(self,
                    outputs: np.array,
                    x: np.array,
                    y: np.array,
                    n_data: int = 2,
                    eta: float = 0.01):
    outputs.insert(0, x)

    N = len(self.layers)
    deltas = []
    delta = None

    loss = np.nan
    for i in range(N, 0, -1):
        output: np.array = outputs[i]
        prev_output: np.array = outputs[i - 1]
        layer: Layer = self.layers[i - 1]

        if i == N:
            d1 = self.dloss(y, output)
            loss = np.sum(d1)
        else:
            layer2: Layer = self.layers[i]
            w2, b2 = layer2.get_weights()
            d1 = w2.dot(delta)

        d2 = layer.dactivation(output).T
        delta = d1 * d2

        delta_w = delta.dot(prev_output).T
        delta_b = delta.reshape([-1])
        deltas.append((delta_w, delta_b))

    layers = self.layers[::-1]
    for i in range(len(deltas) - 1, -1, -1):
        delta_w, delta_b = deltas[i]
        layer = layers[i]

        layer.update_w = 0.5 * layer.update_w + - 2 / n_data * eta * delta_w
        layer.update_b = 0.5 * layer.update_b + - 2 / n_data * eta * delta_b

        layer.w += layer.update_w
        layer.b += layer.update_b

    return dict(loss=loss)
{% endhighlight %}
---
layout: post
title:  "[Optimizer] RMSProp"
date:   2017-04-11 01:00:00
categories: "artificial-intelligence"
asset_path: /assets/posts2/Optimizer/
tags: ['Loss', 'Cost', 'Objective', 'Numpy', 'Scipy']

---

<header>
    <img src="{{ page.asset_path }}rmsprop.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;">
    <small><a href="https://unsplash.com/?photo=iPum7Ket2jo">Matt Palmer의 사진</a>
    </small>
    </div>
</header>


# RMSProp

* Deep Learning 299page - Ian Goodfellow, Yoshua Bengio, and Aaron Courville 참고

RMSProp 알고리즘은 AdaGrad의 gradient accumulation을 exponentially weighted moving average로 변경하여, non-convex 환경에서 더 좋은 performance를 보이도록 수정을 한 알고리즘입니다. (AdaGrad의 경우 convex function에서 빠르게 converge되도록 설계 되어 있습니다.) AdaGrad를 non-convex function에 적용한 경우, learning trajectory는 여러 구조(structure)를 거치다가 궁극적으로 locally convex bowl (local minimum)에 갖히게 되는 현상을 겪을수 있습니다. 이유인즉 AdaGrad의 경우 learning rate가 점차 줄어드는 현상때문입니다. 이때문에 막판에 작은 local minimum에 도달하지만, learning rate가 너무 작아 극복하지 못하고 갖혀버리게 됩니다.

RMSProp의 경우 exponentially decaying average를 사용하여 local minimum에 걸려드는 단점을 극복합니다.(따라서 AdaGrad보다 non-convex에 강함)



### Algorithm

실제 알고리즘은.. AdaDelta와 동일하지만, Update를 계산할때.. 분모부분만 다릅니다.

**Require:** global learning rate $$ \eta $$, decay rate $$ \gamma $$, Epsilon $$ \epsilon $$ (보통 $$ 10^{-8}) $$

#### Pseudo Code

1. Initialize accumulation variables $$ E[g^2] = 0 $$
    1. Compute Gradient $$ g_t $$
    2. Accumulate squared gradient: $$ E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) \odot g^2_t $$
    3. Compute Update: $$ \Delta \theta_t = - \frac{\eta}{\sqrt{\epsilon + E[g^2]_t}} \odot g^2_t $$
    4. Apply Update: $$ \theta_{t + 1} = \theta_t + \Delta \theta_t $$



# Implementation

### Import

{% highlight python %}
%pylab inline
import numpy as np
import pandas as pd

from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
{% endhighlight %}


### Data

{% highlight python %}
iris = load_iris()

# Extract sepal length, petal length from Setosa and Versicolor
data = iris.data[:100, [0, 2]]

# Standardization
scaler = StandardScaler()
data = scaler.fit_transform(data)


# Split data to test and train data
train_x, test_x, train_y, test_y = train_test_split(data, iris.target[:100], test_size=0.3)

# Plotting data
scatter(data[:50, 0], data[:50, 1], label='Setosa')
scatter(data[50:100, 0], data[50:100, 1], label='Versicolour')
title('Iris Data')
xlabel('sepal length')
ylabel('petal length')
grid()
legend()
{% endhighlight %}

<img src="{{ page.asset_path }}iris.png" class="img-responsive img-rounded">




### SGD with RMSProp

{% highlight python %}
w =   np.array([ 0.09370901, -0.24480254, -0.84210235]) # np.random.randn(2 + 1)

def predict(w, x):
    N = len(x)
    yhat = w[1:].dot(x.T) + w[0]
    return yhat

def rmsprop_nn(w, X, Y, eta=0.001, decay=0.001, epoch=4, weight_size=2):
    """
    @param eta <float>: learning rate
    """
    N = len(X)
    e = 1e-8

    Eg = np.zeros(weight_size + 1)  # E[g^2]

    for i in range(N):
        x = X[i]
        y = Y[i]
        x = x.reshape((-1, 2))
        yhat = predict(w, x)

        # Calculate the gradients
        gradient_w = 2/N*-(y-yhat).dot(x)
        gradient_b = 2/N*-(y-yhat)

        # Accumulate Gradient
        Eg[1:] = decay * Eg[1:] + (1-decay) * gradient_w**2
        Eg[0]  = decay * Eg[0] + (1-decay) * gradient_b**2

        # Compute Update
        delta_w = - eta/np.sqrt(e + Eg[1:]) * gradient_w
        delta_b = - eta/np.sqrt(e + Eg[0]) * gradient_b

        w[1:] = w[1:] + delta_w
        w[0] = w[0] + delta_b

    return w


for i in range(90):
    w = rmsprop_nn(w, train_x, train_y)

    # Accuracy Test
    yhats = predict(w, test_x)
    yhats = np.where(yhats >= 0.5, 1, 0)
    accuracy = round(accuracy_score(test_y, yhats), 2)
    print(f'[{i:2}] Accuracy: {accuracy:<4.2}')
{% endhighlight %}


[ 0] Accuracy: 0.0 <br>
[ 1] Accuracy: 0.0 <br>
[ 2] Accuracy: 0.0 <br>
[ 3] Accuracy: 0.03<br>
[ 4] Accuracy: 0.07<br>
[ 5] Accuracy: 0.4 <br>
[ 6] Accuracy: 0.47<br>
[ 7] Accuracy: 0.47<br>
[ 8] Accuracy: 0.47<br>
[ 9] Accuracy: 0.57<br>
[10] Accuracy: 0.6 <br>
[11] Accuracy: 0.7 <br>
[12] Accuracy: 0.73<br>
[13] Accuracy: 0.7 <br>
[14] Accuracy: 0.7 <br>
[15] Accuracy: 0.77<br>
[16] Accuracy: 0.8 <br>
[17] Accuracy: 0.8 <br>
[18] Accuracy: 0.87<br>
[19] Accuracy: 0.87<br>
[20] Accuracy: 0.87<br>
[21] Accuracy: 0.83<br>
[22] Accuracy: 0.87<br>
[23] Accuracy: 0.87<br>
[24] Accuracy: 0.87<br>
[25] Accuracy: 0.87<br>
[26] Accuracy: 0.87<br>
[27] Accuracy: 0.87<br>
[28] Accuracy: 0.9 <br>
[29] Accuracy: 0.9 <br>
[30] Accuracy: 0.9 <br>
[31] Accuracy: 0.93<br>
[32] Accuracy: 0.93<br>
[33] Accuracy: 0.93<br>
[34] Accuracy: 0.93<br>
[35] Accuracy: 0.93<br>
[36] Accuracy: 0.93<br>
[37] Accuracy: 0.93<br>
[38] Accuracy: 0.93<br>
[39] Accuracy: 0.93<br>
[40] Accuracy: 0.93<br>
[41] Accuracy: 0.93<br>
[42] Accuracy: 0.93<br>
[43] Accuracy: 0.93<br>
[44] Accuracy: 0.93<br>
[45] Accuracy: 0.93<br>
[46] Accuracy: 0.93<br>
[47] Accuracy: 0.93<br>
[48] Accuracy: 0.93<br>
[49] Accuracy: 0.93<br>
[50] Accuracy: 0.93<br>
[51] Accuracy: 0.93<br>
[52] Accuracy: 0.93<br>
[53] Accuracy: 0.93<br>
[54] Accuracy: 0.93<br>
[55] Accuracy: 0.93<br>
[56] Accuracy: 0.93<br>
[57] Accuracy: 0.93<br>
[58] Accuracy: 0.93<br>
[59] Accuracy: 0.93<br>
[60] Accuracy: 0.93<br>
[61] Accuracy: 0.93<br>
[62] Accuracy: 0.93<br>
[63] Accuracy: 0.93<br>
[64] Accuracy: 0.93<br>
[65] Accuracy: 0.93<br>
[66] Accuracy: 0.93<br>
[67] Accuracy: 0.93<br>
[68] Accuracy: 0.93<br>
[69] Accuracy: 0.93<br>
[70] Accuracy: 0.93<br>
[71] Accuracy: 0.93<br>
[72] Accuracy: 0.93<br>
[73] Accuracy: 0.93<br>
[74] Accuracy: 0.93<br>
[75] Accuracy: 0.93<br>
[76] Accuracy: 0.93<br>
[77] Accuracy: 0.93<br>
[78] Accuracy: 0.93<br>
[79] Accuracy: 0.93<br>
[80] Accuracy: 0.93<br>
[81] Accuracy: 0.93<br>
[82] Accuracy: 0.93<br>
[83] Accuracy: 0.93<br>
[84] Accuracy: 0.97<br>
[85] Accuracy: 0.97<br>
[86] Accuracy: 1.0 <br>
[87] Accuracy: 1.0 <br>
[88] Accuracy: 1.0 <br>
[89] Accuracy: 1.0 <br>
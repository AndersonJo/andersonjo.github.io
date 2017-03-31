---
layout: post
title:  "Feedforward Neural Network"
date:   2015-09-18 01:00:00
categories: "neural-network"
tags: ['RMS Error', "Hebb's Rule", "아버지 아들 키"]
asset_path: /assets/posts/Feedforward-Neural-Network/
---
<header>
    <img src="{{ page.asset_path }}chappie.jpg" class="img-responsive img-rounded">
</header>

Feedforward는 뉴론간의 연결이 서로 전방향으로만 연결된것을 말합니다.<br>
[Neural Network for Concrete][ref-concrete]를 참고해주세요 :)

먼저 바로 배우기전에 Machine이 어떻게 학습을 하는지 부터 배우도록 하겠습니다.
 

## <span class="glyphicon glyphicon-ok-sign" aria-hidden="true"></span> How a Machine learns

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> Root Mean Square - RMS Error

RMS Method는 예측에 대한 에러률을 알아냅니다.

<img src="{{ page.asset_path }}rms.png" class="img-responsive img-rounded">

예를 들어서.. 아버지키와 아들키에 대한 데이터가 있습니다.

{% highlight r %}
> head(father.son)
   fheight  sheight
1 65.04851 59.77827
2 63.25094 63.21404
3 64.95532 63.34242
4 65.75250 62.79238
5 61.13723 64.28113
6 63.02254 64.24221

> lm(sheight~fheight, father.son)

Call:
lm(formula = sheight ~ fheight, data = father.son)

Coefficients:
(Intercept)      fheight  
    33.8866       0.5141 
    
> father.son$predict <- (father.son$fheight * 0.5141) + 33.8866
> plot(father.son$fheight, father.son$sheight, type='p', col='#333333')
> lines(father.son$fheight, father.son$predict, type='l', col='red')
> sqrt(mean((father.son$predict - father.son$sheight)^2))
[1] 2.434295
{% endhighlight %}

<img src="{{ page.asset_path }}father-son.png" class="img-responsive img-rounded">

눈으로 봐도 아버지키와 아들키사이에는 상관관계가 있음을 알수 있습니다.<br>
중요한것은 이렇게 Linear Regression을 만들었는데 오차가 얼마나 나는가를 측정하는 것입니다.<br>
코드 그리고 공식에서 보듯이 실제 y값과 예측 y값과의 차이입니다.

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> Hebbian Theory

서로 연결된 뉴런들이 동시에 activate된다면 서로 더 연결이 더 강해지지 않을까하는 추측을 50년전 Hebb라는 사람이 생각을 했었고.. 
그 생각을 이론으로 만들어 낸것이  Hebb's Theory입니다. 

이런 문구가 있습니다.  "Neurons that fire together, wire together"
즉 만약 2개의 뉴론이 유사한 activations을 갖고 있다면, 그 둘의 뉴런들의 weights는 더 강하진다는 뜻입니다.

RMS의 경우는 Supervised Learning이었습니다. 즉 예상수치가 있고, 여기에 따라서 수치를 조정해나가는 것입니다.
하지만 Hebb's Rule의 경우는 Unsupervised Learning에 사용됩니다. 


* [Download hebb.py][hebb.py]
* [Reference][ref-hebb]

#### **Formula** ####

**input**값으로 x vector가 있습니다.
<img src="{{ page.asset_path }}Hebb1.png" class="img-responsive img-rounded">

**weight**값으로 -1~1사이의 random 값이 들어가 있는 vector가 있습니다. 
<img src="{{ page.asset_path }}Hebb2.png" class="img-responsive img-rounded">

**output**은 다음과 같이 나옵니다.
<img src="{{ page.asset_path }}Hebb3.png" class="img-responsive img-rounded">

**Delta**값은 다음과 같이 나옵니다.(η은 Learning Rate입니다.)
<img src="{{ page.asset_path }}Hebb4.png" class="img-responsive img-rounded">

예제에서는 단순히 delta값을 weight 값에 더하였습니다.<br>
이렇게하면 weight 값은 점점 극명하게 커지거나 작아지거나 하게 됩니다.<br>
Limit을 두기 위해서는 다음과 같은 Normalization을 할 수 있습니다.


<img src="{{ page.asset_path }}Hebb5.png" class="img-responsive img-rounded">

{% highlight python %}
class Hebb(object):
    def __init__(self):
        # Weights
        self.w1 = 1.
        self.w2 = -1.
        
        # Learning Rate
        self.rate = 1.

        # Epoch Count
        self._epoch_count = 0;

    def process(self):
        for i in range(5):
            self.epoch()

    def epoch(self):
        print '##### Epoch %d ##############' % self._epoch_count
        self.set_pattern(-1., -1.)
        self.set_pattern(-1., 1.)
        self.set_pattern(1., -1.)
        self.set_pattern(1., 1.)
        self._epoch_count += 1

    def set_pattern(self, i1, i2):
        print 'i1=%2d i2=%2d   ' % (i1, i2),
        result = self.recognize(i1, i2)
        print 'result=%5d    ' % result,

        delta = self.train(self.rate, i1, result)
        self.w1 += delta
        print 'delta1=%4d  ' % delta,

        delta = self.train(self.rate, i2, result)
        self.w2 += delta
        print 'delta2=%4d    ' % delta,
        print 'w1=%d   w2=%d'% (self.w1, self.w2),
        print

    def recognize(self, i1, i2):
        return (self.w1 * i1 + self.w2 * i2) * 0.5

    def train(self, rate, input, output):
        return rate * input * output
{% endhighlight %}

{% highlight python %}
##### Epoch 0 ##############
i1=-1 i2=-1    result=    0     delta1=   0   delta2=   0     w1=1   w2=-1
i1=-1 i2= 1    result=   -1     delta1=   1   delta2=  -1     w1=2   w2=-2
i1= 1 i2=-1    result=    2     delta1=   2   delta2=  -2     w1=4   w2=-4
i1= 1 i2= 1    result=    0     delta1=   0   delta2=   0     w1=4   w2=-4

##### Epoch 4 ##############
i1=-1 i2=-1    result=    0     delta1=   0   delta2=   0     w1=256   w2=-256
i1=-1 i2= 1    result= -256     delta1= 256   delta2=-256     w1=512   w2=-512
i1= 1 i2=-1    result=  512     delta1= 512   delta2=-512     w1=1024   w2=-1024
i1= 1 i2= 1    result=    0     delta1=   0   delta2=   0     w1=1024   w2=-1024
{% endhighlight %}


기본값으로 w1=1, w2=-1 이 주어집니다.<br>
Epoch를 여러번 돌면서 서로 관련이 있는 뉴런들을 강화시켜줍니다.

recognize 메소드는 weights에 따른 뉴런의 예측값을 연산합니다.<br>
weight1 * input 1 은 노드 1번의 결과값을 result 노드로 보냅니다.<br>
weight2 * input 2 는 노드 2번의 결과값을 result 노드로 보냅니다.<br>
result노드는 2개의 노드에서 받은 값을 합치고 weight값인 0.5로 곱해서 최종 예측값을 도출합니다. 

train 메소드가 Hebb's Rule 공식부분입니다.<br>
Learning Rate * Input Value * Anticipated Result<br>
recognize로 나온 예측값에 Input Value를 곱해준다는 뜻은 원래의 추세방향을 더욱 강화시켜주겠다는 뜻입니다.<br>

간단하게 설명하면.. 어떤 뉴런 활동을 반복하면 할수록 서로 동일한 뉴런들일수록 더욱 강해집니다.
 
그러면 어떤 뉴런이 Positive 강해지고, 어떤 뉴런이 Negative 로 강하질까요?

**(WeightA * InputA) 그리고 (WeightB * InputB) 의 연산을 하게 되는데 <br>
이때 서로의 값이 동일하게 positive이거나 negative에 따라서 결정이 됩니다.**

예를 들어서 

WeightA(1) * InputA(-1) 그리고 WeightB(-1) * InputB(1) = **-1 그리고 -1**<br>
즉 둘다 음수이기 때문에 음수로 강해지게 되고, 

WeightA(1) * InputA(1) 그리고 Weight(-1) * InputB(-1) = **1 그리고 1**<br>
즉 둘다 양수이기 때문에 양수로 강해지게 됩니다.


## <span class="glyphicon glyphicon-ok-sign" aria-hidden="true"></span> Feedforward Neural Network

* [xor.py][xor.py]
* [activation.py][activation.py]
* [feedforward.py][feedforward.py]

Feedforward Connection은 [Concrete Strength][ref-concrete]처럼 전방향으로만 레이어간에 연결이 되어진 모델의 뉴럴 네트워크입니다.<br>
특징으로는..

* 가장 단순한 형태의 첫번째 뉴럴 네트워크 
* Cycle 또는 loop이 없는 형태
* recurrent neural network와는 다름
* 실무에서 쓰일수 있는 가장 단순한 형태의 뉴럴 네트워크
* Multi-Layer Perceptron 으로도 불림




[hebb.py]: {{ page.asset_path }}hebb.py
[xor.py]: {{ page.asset_path }}xor.py
[activation.py]: {{ page.asset_path }}activation.py
[feedforward.py]: {{ page.asset_path }}feedforward.py

[ref-concrete]: /neural-network/2015/07/26/Neural-Network-for-concrete/
[ref-hebb]: https://wiki.eyewire.org/en/Hebb's_rule

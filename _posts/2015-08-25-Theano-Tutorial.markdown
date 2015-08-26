---
layout: post
title:  "Theano Tutorial"
date:   2015-08-25 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Theano-Tutorial/
---
<div>
    <img src="{{ page.asset_path }}deep-learning.jpg" class="img-responsive img-rounded">
</div>

Deep Learning 을 구현하는데에 지원하는 많은 라이브러리및 프레임워크가 있습니다.
DL4j, Caffe, Torch등등 많이 있지만.. 오늘 제가 소개해드릴 라이브러리는 Theano 라고 하는 강력한 툴입니다.

Theano의 최대 장점은 바로 GPU및 C언어 레벨에서의 연산처리에 있습니다. 즉 CUDA C를 집접 작성할 필요가 없고 또한 C언어를 집접 코딩할 필요도 없습니다.
Theano가 GPU및 C언어 및 GPU 레벨단에서의 처리를 자동으로 처리하기 때문이죠. 그러면서도 가장 인간친화적인 Python언어를 사용할수 있다는 것입니다.

수학적 공식부터 차근차근 Machine Learning을 공부하는 분들부터, 실제 퍼포먼스를 중요하게 따지는 Production에서도
문제없이 사용이 가능합니다.

## 01 - Scalar and Function

[Download tutorial_sin.py][tutorial_sin]

Trigonometric Functions 중에 다음과 같은 공식이 있습니다. 

> <img src="{{ page.asset_path }}sin-cos.gif" class="img-responsive img-rounded">

이 공식을 Theano 에서 만들고 실행시켜보겠습니다. 

{% highlight python %}
from theano import tensor as T
from theano import function

"""
Example 01 : Sin
"""
x = T.dscalar('x')
f = function([x], T.sin(T.deg2rad(x)) ** 2 + T.cos(T.deg2rad(x)) ** 2)
print f(0)  # 1.0
print f(30)  # 1.0
print f(45)  # 1.0
print f(60)  # 1.0
print f(90)  # 1.0

{% endhighlight %}

뭐 대단한게 아니네 라고 느끼실지 모르겠지만.. Theano가 대단한것은 바로 C언어로 Dynamic Compile을 해주기 때문입니다.
실제로 프로그램을 처음실행시 처리 속도가 늦습니다. 이유는 C언어로 Compile을 하고 cache시키기 때문입니다. 
따라서 f(x) 함수를 실행시 파이썬에서 돌아가는 것이 아니라 C언어에서 빠르게 돌아갑니다. (또는 GPU)

위에서부터 코드를 보면.. T.dscalar가 있는데 double scalar라는 뜻이고, scalar는 숫자값을 의미합니다.<br>
function함수를 통해서 input값과 처리할 수학적 공식을 output 으로 적습니다. 

## 02 - Shared Variable

### Shared Variable 

[Download tutorial_shared_variable.py][tutorial_shared_variable]

Shared Variable 은 함수들 사이에서 공유가 될수 있습니다.

{% highlight python %}
from theano import tensor as T
from theano import function
from theano import shared

state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state + inc)])
decrementor = function([inc], state, updates=[(state, state - inc)])

accumulator(1)  # return value: 0
print 'state:', state.get_value()  # state: 1

accumulator(3)  # return value: 1
print 'state:', state.get_value()  # state: 4

decrementor(1)  # return value: 4
print 'state:', state.get_value()  # state: 3

accumulator(1)  # return value: 3
state.set_value(10)
print 'state:', state.get_value()  # state: 10

{% endhighlight %}

예제에서는 accumulator 함수와 decrementor 함수 두군데에서 shared variable이 사용이 되고 있습니다.<br>
.get_value, .set_value 함수를 통해서 shared variable의 값을 설정하거나 꺼내올수 있습니다.<br>

updates parameter가 있는데 여기에는 형식으로 list of pairs of the form [(shared variable, expression)] 반드시 적어주어야 합니다.

shared variable을 사용하는 가장 주된 목적은 효율성에 있습니다. 즉 Theano는 어디에다가 shared variable을 할당할 것인지 컨트롤이 가능합니다.
그리고 GPU안에다가 할당시 효율성을 더욱 극대화 시킬수 있습니다.


### Given Parameter

Given parameter를 이용해서 특정 constant, shared variable, etc 등을 **replace** 시킬수 있습니다.

{% highlight python %}

foo = T.scalar(dtype=state.dtype)
fn_of_state = state * 2 + inc
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print skip_shared(1, 3)  # 7
print state.get_value()  # 10

{% endhighlight %}

fn_of_state 라는 expression안의 state를 foo로 바꾸는 것입니다. 단순히 replace라고 생각하시면 편합니다.


## 03 - Random Number

[Download tutorial_shared_variable.py][tutorial_random]

Theano에서는 어떠한 공식이나 수식을 symbol을 사용해서 compile을 하기 때문에, Numpy 처럼 random을 사용하는게 쉽지 않습니다.
randomness를 구현하기 위해서는 symbol처럼 random variable을 수식(expression)에 넣어주는게 좋은 방법입니다.

{% highlight python %}
from theano.tensor.shared_randomstreams import RandomStreams

random_stream = RandomStreams(seed=1234)
r_uniform = random_stream.uniform(size=(2, 2))
r_normal = random_stream.normal(size=(2, 2))

f = function([], r_uniform)
g = function([], r_normal, no_default_updates=True)

print f()  # [[ 0.75718064  0.1130526 ] [ 0.00607781  0.8721389 ]]
print f()  # [[ 0.31027075  0.24150866] [ 0.56740797  0.73226671]]

print g()  # [[ 0.42742609  1.74049825] [-0.02169041  1.48401086]]
print g()  # [[ 0.42742609  1.74049825] [-0.02169041  1.48401086]]

random_stream.seed(7777)
print g()  # [[-0.28277921 -0.12554144] [ 1.56899783 -1.10901327]]

{% endhighlight %}

Theano 는 random generator를 만들기 위해서 Numpy RandomStream object 를 사용합니다.<br>
r_uniform 은 여기에서 2*2 matrix 를 나타내며 uniform은 확률상 일정하게 랜덤값이 나오며, normal은 normal distribution에 따라서
(68%의 값들이 mean값에서 one standard deviation에 포함되는 그래프) 랜덤값이 나오게 됩니다.<br>

no_default_updates=True 값을 하면 g()함수를 불렀을때 random state값이 변경이 안되서, g()함수를 부를때마다 동일한 결과값을 얻습니다.

주의하실 점은 **Random Stream은 오직 CPU에서만** 작동을 하게 됩니다. (GPU방법은 따로 있습니다.)

## 04 - Recognizing Hand-written Numbers

* [Download mnist.pkl.gz][mnist]
* [Reference web page][deep-learning-get-started]

<img src="{{ page.asset_path }}small_mnist.png" class="img-responsive img-rounded">

MNIST은 집접 손으로 쓴 숫자 이미지가 담긴 dataset 이다. 트레이닝용으로 60,000개의 이미지가 있으며, 10,000개의 테스팅용 이미지가 있습니다.
트레이닝셋 60,000개는 실질적인 트레이닝용으로 50,000개로 그리고 10,000개의 Validation용으로 사용이 됩니다.
모든 디지털 이미지는 size-normalized되어서 **28 x 28**의 사이즈를 갖으며, 각 픽셀당 나타내는 숫자값은 0~255를 갖습니다. 
0은 black 이며, 255은 white 색입니다.

### Load mnist.pkl.gz
{% highlight python %}
import cPickle, gzip

# Load the dataset
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)
{% endhighlight %}






[tutorial_sin]: {{page.asset_path}}tutorial_sin.py
[tutorial_shared_variable]: {{page.asset_path}}tutorial_shared_variable.py
[tutorial_random]: {{page.asset_path}}tutorial_random.py
[deep-learning-get-started]: http://deeplearning.net/tutorial/gettingstarted.html
[mnist]: http://deeplearning.net/data/mnist/mnist.pkl.gz
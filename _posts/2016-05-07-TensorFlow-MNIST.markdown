---
layout: post
title:  "TensorFlow - Softmax Regression"
date:   2016-05-07 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/TensorFlow-Softmax-Regression/
tags: ['MNIST', 'Logistic', 'Sigmoid', 'binary', 'partial derivative', 'odds ratio']

---

<header>
    <img src="{{ page.asset_path }}digits.png" class="img-responsive img-rounded">
</header>

# MNIST

### MNIST Dataset

[MNIST Website][MNIST Website]에서 MNIST 데이터를 다운받을수 있습니다.

다운로드된 파일은 다음과 같이 구성되어 있습니다.

* **train-images-idx3-ubyte.gz**:  training set images (9912422 bytes) 
* **train-labels-idx1-ubyte.gz**:  training set labels (28881 bytes) 
* **t10k-images-idx3-ubyte.gz**:   test set images (1648877 bytes) 
* **t10k-labels-idx1-ubyte.gz**:   test set labels (4542 bytes)

파이썬으로 집접 보기 위해서는 gzip, struct 를 사용해서 풀면됩니다. 

### MNIST In TensorFlow

MNIST는 [MNIST Website][MNIST Website]에서 다운 받을수 있지만, <br> 
TensorFlow는 이미 자동으로 다운받을수 있는 코드를 포함하고 있습니다.


{% highlight python %}
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
{% endhighlight %}


다운로드된 데이터는 3가지로 분류될수 있습니다.

* **mnist.train** 55,000개의 트레이닝 데이터 
* **mnist.test** 10,000개의 테스트 데이터 
* **mnist.validation** 5,000개의 검증 데이터 

 이미지 샘플을 보기 위해서는 다음과 같은 코드를 하면 됩니다.

{% highlight python %}
fig, subplots = pylab.subplots(8, 10) # subplots(y축, x축 갯수)

idx = 0
for _subs in subplots:
    for subplot in _subs:
        d = mnist.train.images[idx]
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        subplot.imshow(d.reshape(28, 28), cmap=cm.gray_r)
        idx += 1
{% endhighlight %}
 
<img src="{{ page.asset_path }}sample.png" class="img-responsive img-rounded">


# Softmax Regression

Logistic Regression 에서는 labels이 binary: $$y^{(i)} \in \{0,1\}$$ 입니다.
하지만 Softmax Regression (or multinomial logistic regression) 에서는 2개 이상의 
mutually exclusive classes를 분류할때 $$y^{(i)} \in \{1,\ldots,K\}$$ 사용합니다.
즉 0~9까지의 클래스들이 있으며 서로의 숫자는 mutually exclusive이기 때문에 Softmax Regression이 좋은 알고리즘이 될 수 있습니다.

Softmax Regression은 2가지 파트로 구성이 되어있습니다.<br>
특정 classes (여기서는 0~9) 안에 들어가는지 weights 값과 inputs값을 계산해서 evidence를 이끌어내는 부분과,<br>
만들어진 evidence를 link function을 이용해서 확률로 만들어주는 것입니다.

$$ evidence_{i} = \sum_{j}{W_{i,j} x_{j} + b_{i}} $$

* $$ W_{i} $$ 는 weights. (784 * 10 matrix)
* $$ b_{i} $$ 특정 클래스 i의 bias값 (i는 픽셀당 10개가 존재, j는 픽셀갯수 784)

evidence가 나오면 해당값을 softmax function을 이용해서 확률로 변환해줍니다.<br>
즉 softmax function은 여기에서 activation function 또는 link function의 역활을 해줍니다.

$$ y = softmax(evidence) = normalize(exp(evidence)) = \frac{exp(evidence_{i})}{\sum_{j}{exp(evidence_{j})}} $$

예를 들어서 다음과 같은 Evidence(logits)가 있습니다. 
exp(logits)을 해주면 $$ e^{0} $$ 이므로 모두 1값으로 변합니다.
이후 sum을 해주는데 axis를 1로 잡을때 0으로 잡을때의 차이를 아는것이 중요합니다.

{% highlight python %}
logits
# array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

np.exp(logits) 
# array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
#        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])

np.sum(np.exp(logits), 1) # right
# array([ 10.,  10.])

np.sum(np.exp(logits), 0) # wrong
# array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
{% endhighlight %}     




좀 더 쉽게 설명한 이미지.. <br>
softmax는 link function의 역활을 하면서, 각각의 양의 수치들을 확률로 변경해줍니다.

<img src="{{ page.asset_path }}udacity_logistic01.png" class="img-responsive img-rounded">

<img src="{{ page.asset_path }}udacity_logistic02.png" class="img-responsive img-rounded">

**Numpy**

{% highlight python %}
def cal_net_input(x, w=np.zeros([784, 10]), bias=np.zeros(10)):
    """
    x: 784개의 arrays를 갖은 array
        ex) [[ 0.  0.  0. ...,  0.  0.  0.] <- 784 array
             [ 0.  0.  0. ...,  0.  0.  0.]]
    """
    # Make Evidence
    # y값은 다음과 같이 나옵니다.
    # [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    y = np.matmul(x, w) + bias
    return y

def cal_softmax(logits):
    """
    np.exp(logits) -> [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.] 
                       [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
    np.sum(np.exp(logits), axis=1)) -> [ 10.  10.]
    np.exp(logits).T -> [[ 1.  1.]
                         [ 1.  1.]
                         [ 1.  1.]
                         [ 1.  1.]
                         [ 1.  1.]
                         [ 1.  1.]
                         [ 1.  1.]
                         [ 1.  1.]
                         [ 1.  1.]
                         [ 1.  1.]]
    """
    # Softmax Regression or Normalization
    return (np.exp(logits).T / np.sum(np.exp(logits), axis=1)).T

batch_xs, batch_ys = mnist.train.next_batch(2)
predicted_y = cal_softmax(cal_net_input(batch_xs))
# [[ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
#  [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]]
{% endhighlight %}

참고로 cal_softmax의 로직은 Tensorflow에 맞춘 것입니다..<br>
구글 딥러닝 자료에 보면 다음과 같이 코딩 하였습니다. (차이가 나는 이유는 구글 딥러닝 코스에서는 1차원 array를 사용했기 때문으로 생각됨)

{% highlight python %}
def cal_softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=0)
{% endhighlight %}


**TensorFlow**

{% highlight python %}
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
evidence = tf.matmul(x, w) + b
y = tf.nn.softmax(evidence)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})
    # [[ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
    #  [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]]
{% endhighlight %}

# Training 

머신러닝에서는 뭐가 좋은 모델인지 알기 위해서 cost function 또는 loss function을 사용합니다. <br>
즉 결과치가 얼마나 안 좋은지를 수치적으로 판단하는 것이죠. <br>
그중에서 **Cross-entropy cost function**을 사용해볼 수 있습니다. 

$$ H_{y'}(y) = -\sum_i y'_i \log(y_i) $$

**Numpy**

{% highlight python %}
def cal_cross_entropy(predicted_y, y_):
    """
    predicted_y: n * 10 arrays 
        0~9까지의 각각의 확률을 갖고 있는 예측값
        ex) [[ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1],
             [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]]
                   
    y_: n * 10 arrays를 갖고있으며, 실제 정답이 되는 값을 갖고 있다.
        ex) [[ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]]
    
    주의할점은 y_ * log(y) 에서 dot을 쓰는게 아니라 그냥 multiplication을 해준다.
    """
    return np.mean(-np.sum(y_ * np.log(predicted_y), 1))
      
cross_entropy_data = cal_cross_entropy(predicted_y, batch_ys)
# 2.30258509299
{% endhighlight %}


**TensorFlow**

{% highlight python %}
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
... 생략 ...

sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
# 2.30259
{% endhighlight %}


트레이닝을 시켜보도록 하겠습니다.<br>
mnist.train.next_batch(100) 는 100개의 **random**데이터를 가져오게 됩니다. (순서대로 가져오는게 아닙니다.)<br>
weights, bias값들이 업데이터가 되고 evidence로 예측을 해보면 숫자 1일때의 확률이 8.19로 가장 높게 나오는것을 확인할수 있습니다.

{% highlight python %}
... 생략 ...
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
    print batch_ys
    print sess.run(evidence, feed_dict={x: batch_xs})

# [[ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]]
# [[-5.03200531  8.19258213  2.96868229  1.53522527 -5.14560604 -0.55663192
#   -1.27567005 -2.03583741  2.86174178 -1.51246953]]
{% endhighlight %}



[MNIST Website]: http://yann.lecun.com/exdb/mnist/
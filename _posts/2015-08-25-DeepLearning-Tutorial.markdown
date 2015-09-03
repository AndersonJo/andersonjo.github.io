---
layout: post
title:  "DeepLearning Tutorial"
date:   2015-09-03 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/DeepLearning-Tutorial/
---
<div>
    <img src="{{ page.asset_path }}neural.jpg" class="img-responsive img-rounded">
</div>

{% comment %}

<embed src="http://player.bgmstore.net/Sldcr" allowscriptaccess="always" allowfullscreen="true" width="422" height="180" /><br>
<a href="http://bgmstore.net/view/Sldcr" target="_blank">BGM정보 : 브금저장소 - http://bgmstore.net/view/Sldcr</a>

쓸때없이 브금 넣어봄 ;;; ㅋㅋㅋㅋ

{% endcomment %}

## Recognizing Hand-written Numbers

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
[gd-wiki]: https://en.wikipedia.org/wiki/Gradient_descent
[gd-data]: {{page.asset_path}}data.csv
[gd-py]: {{page.asset_path}}tutorial_gradient.py

[mnist]: http://deeplearning.net/data/mnist/mnist.pkl.gz
---
layout: post
title:  "TensorFlow - TensorBoard"
date:   2016-06-23 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/TensorBoard/
tags: []

---

<header>
    <img src="{{ page.asset_path }}graph_vis_animation.gif" class="img-responsive img-rounded">
</header>

# Serializing the data

TensorBoard는 summary data를 갖고 있는 TensorFlow events files 을 읽음으로서 작동이 됩니다.
먼저 summary data를 얻고자 하는 TensorFlow graph를 만들고, 이후 어떤 nodes를 [summary_operations][summary-operations]에 넣을지 결정합니다.

예를 들어서, Convolutional Neural Network를 사용해 MNIST digits을 분석하려고 한다면, 
learning rate를 시간에 따라서 기록하거나, 어떻게 objective function이 변화하는지 알아볼수 있습니다.
[scalar-summary][scalar-summary]를 통해서 이러한 값들을 붙여주고, 이름을 *learning rate* 또는 *loss function*같은 이름으로 태그를 주면 됩니다.


# Running TensorBoard

실행시킬때는 반드시 logdir을 맞춰줍니다.

{% highlight python %}
tensorboard --logdir /tmp/mnist_logs/
{% endhighlight %}
 
[summary-operations]: https://www.tensorflow.org/versions/r0.9/api_docs/python/train.html#summary-operations
[scalar-summary]: https://www.tensorflow.org/versions/r0.9/api_docs/python/train.html#scalar_summary
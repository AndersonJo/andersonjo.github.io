---
layout: post
title:  "TensorFlow And Numpy"
date:   2016-05-06 01:00:00
categories: "machine-learning"
static: /assets/posts/Perceptron/
tags: ['python', 'data analytics', 'jupyter', 'ipython', 'notebook']
---

### Matrix Indexing 

**Numpy**

{% highlight python %}
d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
d[[0, 1, 2], [2, 1, 0]]
# array([3, 5, 7])
{% endhighlight %}

**TensorFlow**

{% highlight python %}
x = tf.Variable([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    print sess.run([tf.gather_nd(x, [[0, 2], [1, 1], [2, 0]])])[0]
    # [3 5 7]
{% endhighlight %}
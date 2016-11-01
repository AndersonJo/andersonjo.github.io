---
layout: post
title:  "TensorFlow - Convolutional Neural Networks Part 2"
date:   2016-06-13 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Convolutional-Neural-Networks/
tags: ['CNN', 'Memory']

---

<header>
    <img src="{{ page.asset_path }}R6S_Screenshot_shield.jpg" class="img-responsive img-rounded" style="width:100%">
</header>

# Useful Tips

### Array to Image in Ipython

Numpy Array를 이미지로 출력시킵니다.

{% highlight python %}
from IPython.display import display
from scipy.misc import toimage

display(toimage(image_array))
{% endhighlight %}


# Convolutional Neural Network with TFLearn

### 메모리 제한

TensorFlow의 Session은 기본적으로 GPU의 모든 자원을 다 할당을 받습니다.<br>
특정 메모리를 할당받기 위해서는 다음과 같이 합니다.

{% highlight python %}
tflearn.config.init_graph(gpu_memory_fraction=0.4)
{% endhighlight %}


### Retrieve CIFAR-10

{% highlight python %}
# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)
{% endhighlight %}

### Data Visualization

{% highlight python %}
fig, subplots = pylab.subplots(10, 15) # subplots(y축, x축 갯수)

idx = 10
for _subs in subplots:
    for subplot in _subs:
        d = X[idx]
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        subplot.imshow(d, cmap=cm.gray_r)
        idx += 1
{% endhighlight %}

<img src="{{ page.asset_path }}cifar-10-visualization.png" class="img-responsive img-rounded" >

### Data PreProcessing and Augmentation

{% highlight python %}
# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
{% endhighlight %}

### Building Convolutional Network

{% highlight python %}
# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 128, 4, activation='relu')
network = conv_2d(network, 128, 4, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = fully_connected(network, 512, activation='relu')
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
{% endhighlight %}

### Create a Model

{% highlight python %}
model = tflearn.DNN(network, tensorboard_verbose=0)
{% endhighlight %}

### Training

{% highlight python %}
# Train using classifier
model.fit(X, Y, n_epoch=80, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='cifar10_cnn')
{% endhighlight %}

### Save 

{% highlight python %}
model.save('cifar10.tfmodel')
{% endhighlight %}
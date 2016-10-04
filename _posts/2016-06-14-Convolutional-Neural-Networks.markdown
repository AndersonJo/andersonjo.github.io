---
layout: post
title:  "TensorFlow - Convolutional Neural Networks"
date:   2016-06-14 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Convolutional-Neural-Networks/
tags: ['CNN', 'Gimp', 'Nvidia', 'GPU Memory']

---

<header>
    <img src="{{ page.asset_path }}R6S_Screenshot_shield.jpg" class="img-responsive img-rounded" style="width:100%">
</header>

# Convolutional Neural Networks

### What is Convolution?

<img src="{{ page.asset_path }}Convolution_schematic.gif" class="img-responsive img-rounded">

왼족의 matrix를 흑백을 나타내는 이미지라고 합니다. (0은 검정색, 1은 흰색) <br>
3 by 3으로 움직이는 sliding window는 kernel, filter 또는 feature detector 등으로 불립니다. <br>
각각 element-wise 로 곱셉을 해준뒤, 합계를 오른쪽에다 써주게 됩니다.

직관적으로 이해하기 위해서는 Ubuntu의 Gimp를 사용해서 알아볼수 있습니다. <br>
**(Gimp -> Filters -> Generic -> Convolution Matrix)** 

<img src="{{ page.asset_path }}rainbowsix_siege_convolutioned.png" class="img-responsive img-rounded">

간단히 말해서 이미지의 한 부분이 서로 비슷비슷한 색상을 갖고 있으면 서로 상쇄를 시켜줘서 0값이 되지만, 
갑작이 sharp-edge 부분을 만나게 되면은 색상차이가 갑자기 커져서 해당 부분의 convolution값이 높아지게 됩니다.
위의 예제에서처럼 Rainbow Six Siege의 slege의 사진중에 윤곽선만 도드라져 보이는게 보일것입니다. 이러한 이유때문에 저렇게 나오는 것입니다.
  

### What are convolutional Neural Networks?

Convolution의 개념에 대해서 알았다면, CNN 이란 질문이 나올 것 입니다.<br>
CNN은 기본적으로 several layers of convolutions이 결과값에 ReLU, Tanh같은 Nonlinear function을 적용한 것들입니다.
먼저 input image에 convolution을 적용하여 output을 꺼냅니다. 해당 output은 input image의 local connection으로 연결이 되어 있으며, 
각각의 layers들은 각기 다른 filters를 적용합니다.

**궁극적으로 CNN은 filters의 값들을 자동으로 학습을 합니다**. (그게 뭐건간에 내가 다루는 task에 따라서 자동으로)<br>
예를 들어서 Image Classification에서 첫번째 layer에서 edges들을 인식할것입니다.<br>
그뒤 해당 edges들을 사용하여, 간단항 형태(shapes)들을 2번째 layer에서 인식을 하게 됩니다.<br>
그뒤 해당 simple shapes들을 사용하여, 더 높은 형태의 특징들 (예를 들어서 얼굴, 사과, 동전, 자동차 등등..)의 특징들을 인식하게 됩니다.<br>
마지막 layer에서는 더 높은 차원의 특징들을 뽑아내 classification을 하게 됩니다.


<img src="{{ page.asset_path }}cnn_architecture.png" class="img-responsive img-rounded">


### Check My GPU Systems

{% highlight bash %}
$ nvidia-smi       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 367.44                 Driver Version: 367.44                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1070    Off  | 0000:01:00.0     Off |                  N/A |
|  0%   57C    P0    41W / 230W |    799MiB /  8112MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      1480    G   compiz                                         233MiB |
|    0      4335    G   ...s-passed-by-fd --v8-snapshot-passed-by-fd   117MiB |
|    0      5975    G   ...ps/pycharm-2016.2/bin/../jre/jre/bin/java    18MiB |
+-----------------------------------------------------------------------------+
{% endhighlight %}


### References

[WildML Convolutional Neural Network][WildML Convolutional Neural Network]<br>
Thanks for sharing awesome deep learning tutorials! Thanks! 





[WildML Convolutional Neural Network]: http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
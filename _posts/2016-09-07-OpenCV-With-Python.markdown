---
layout: post
title:  "OpenCV with Python"
date:   2016-08-31 01:00:00
categories: "vision"
asset_path: /assets/posts2/OpenCV/
tags: ['Python']

---

<header>
    <img src="{{ page.asset_path }}walle.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small>
       몇년전 FFMPEG으로 Transcoding할때 OpenCV를 사용하게 되었습니다. 그때 Face Detection을 했었는데 (정말 기초부분이죠..) 그거 하고나서 와~ 하는 감탄사가 ㅋ <br>
       Python으로 정말 쉽게 OpenCV를 다루는 법을 공유하고자 합니다. <br>
    </small>
    </div>
</header>

# Installation on Ubuntu

### Using the Ubuntu Repository (no support for depth camera)

{% highlight bash %}
sudo apt-get update
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
sudo apt-get install libopencv-*
sudo apt-get install python-opencv
{% endhighlight %}

### Using the CMkae and Compiler

Depth camera를 지원하기 위해서는 (Kinect를 포함) OpenNI 그리고 SensorKinect를 설치해야 합니다. 

{% highlight bash %}
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install ffmpeg
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
{% endhighlight %}

**OpenNI 설치**

먼저 OpenNI dependencies를 설치합니다.

{% highlight bash %}
sudo apt-get install libusb-1.0-0-dev 
{% endhighlight %}

[OpenNI][OpenNI] 에 들어가서 다운을 받습니다.<br>

{% highlight bash %}
tar -xvf OpenNI-Linux-x64-2.2.0.33.tar.bz2
cd OpenNI-Linux-x64-2.2/
sudo chmod a+x install.sh
sudo ./install.sh
{% endhighlight %}


**SeonsorKinect 설치**
 
[OpenCV 3.1.0][OpenCV 3.1.0]를 다운받습니다. 또는 [OpenCV Download Page][OpenCV Download Page]에서 원하는 버젼을 다운받습니다. 

{% highlight bash %}
unzip opencv-3.1.0.zip
cd opencv-3.1.0
mkdir build
cd build
{% endhighlight %}

{% highlight bash %}
cmake-gui
{% endhighlight %}

또는 집접 command 를 사용해서 build를 합니다.

{% highlight bash %}
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ../
{% endhighlight %}


# Tutorial 

### Image IO

**Read Image as BGR Color**
{% highlight python %}
%pylab inline
import numpy as np
import cv2
print "OpenCV Version : %s " % cv2.__version__
{% endhighlight %}

{% highlight python %}
img = cv2.imread('../data/images/baseball.jpg', cv2.CV_LOAD_IMAGE_COLOR)
pylab.imshow(img)
{% endhighlight %}

OpenCV에서는 BGR(RGB의 반대)로 읽기 때문에, cv2.imread할때 제대로 읽어와진것은 맞으나, Pylab에서 이미지를 띄울때 잘못된 색상으로 보여줍니다.

<img src="{{ page.asset_path }}inverted_baseball.png" class="img-responsive img-rounded">

**Convert BGR Image to RGB Image**
{% highlight python %}
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pylab.imshow(img_rgb)
{% endhighlight %}

<img src="{{ page.asset_path }}rgb_baseball.png" class="img-responsive img-rounded">

**Write an image as different format**


### Spliting Image channels

split 함수로 b, g, r 각각 따로따로 channel을 나눌수 있고, merge를 통해서 channel을 하나로 합칠수 있습니다.

{% highlight python %}
b1, g1, r1 = cv2.split(img1)
b2, g2, r2 = cv2.split(img2)
img3 = cv2.merge((b2, g1, r2))
{% endhighlight %}

<img src="{{ page.asset_path }}mixed_channels.png" class="img-responsive img-rounded">


### Blending two images

addWeightedb() 함수는 다음과 같은 공식을 따릅니다.

$$ g(x) = (1 - \alpha)f_0(x) + \alpha f_1(x)$$ 

{% highlight  python %}
img3 = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
pylab.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
{% endhighlight %}

<img src="{{ page.asset_path }}blended_baseball.png" class="img-responsive img-rounded">
<small>근데 이거 사진이 너무 유치한데.. </small>

[OpenNI]: http://structure.io/openni
[OpenCV 3.1.0]: https://github.com/Itseez/opencv/archive/3.1.0.zip
[OpenCV Download Page]: http://opencv.org/downloads.html

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


[OpenNI]: http://structure.io/openni
[OpenCV 3.1.0]: https://github.com/Itseez/opencv/archive/3.1.0.zip
[OpenCV Download Page]: http://opencv.org/downloads.html

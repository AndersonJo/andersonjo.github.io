---
layout: post
title:  "OpenCV with Python"
date:   2016-09-07 01:00:00
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

apt-get으로 python-opencv를 설치시 대략 2.x 버젼이 설치가 됩니다. (필자의 경우.. 2.4.9.1)

{% highlight bash %}
sudo apt-get update
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
sudo apt-get install libopencv-*
sudo apt-get install python-opencv
{% endhighlight %}

### Installing OpenCV 3 from Source on Ubuntu

먼저 dependencies롤 모두 설치해줍니다. 아래의 라이브러리는 이미지, 비디오, optimizaiton등을 위해서 필요한 라이브러리들입니다.

{% highlight bash %}
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake git pkg-config
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python2.7-dev
sudo apt-get -y install libopencv-dev build-essential cmake git libgtk2.0-dev pkg-config python-dev python-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip
{% endhighlight %}

그 다음은 Source에서 설치하기 위해서 git에서 clone합니다. (이거 졸라 개 오 래 걸 림) 

{% highlight bash %}
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 3.1.0
{% endhighlight %}


Makefile을 만들기 위해서 cmake를 합니다.

{% highlight bash %}
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D BUILD_opencv_java=ON \
	-D WITH_V4L=ON \
	-D WITH_TBB=ON \
	-D WITH_OPENGL=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D WITH_CUBLAS=1 \
	-D WITH_FFMPEG=ON \
	-D BUILD_EXAMPLES=ON .. 
{% endhighlight %}

build를 합니다. (이때 -j8 같은 옵션을 줄 경우 CPU cores 8개를 사용하서 make를 합니다.)

{% highlight bash %}
make -j8
sudo make install
sudo ldconfig
{% endhighlight %}



# Basic Tutorial 

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


### Video Capture

{% highlight python %}
cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert to RGB
    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#     pylab.imshow(color_frame)
#     pylab.show()
{% endhighlight %}

<img src="{{ page.asset_path }}video_capture.png" class="img-responsive img-rounded">
<small>아~~~~ 나도 좀 잘생겨 보고 싶다!!!!!!!!!!! 죄송죄송~</small>


# SIFT (Scale-Invariant Feature Transform)



### References 

* [SIFT Explained](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/)



[OpenNI]: http://structure.io/openni
[OpenCV 3.1.0]: https://github.com/Itseez/opencv/archive/3.1.0.zip
[OpenCV Download Page]: http://opencv.org/downloads.html

---
layout: post
title:  "CUDA Toolkit, cuDNN, TensorFlow 101"
date:   2016-05-04 01:00:00
categories: "machine-learning"
static: /assets/posts/TensorFlow-101/
tags: ['CUDA', 'GTX960', 'Nvidia', 'Ubuntu', 'format']

---

<header>
<img src="{{ page.static }}tensorflow.jpg" class="img-responsive img-rounded" style="width:100%">
</header>


# Google DeepMind uses it!

구글 딥마인드는 Torch7에서 TensorFlow로 갈아타기 시작했습니다.

[DeepMind moves to TensorFlow][DeepMind moves to TensorFlow]<br>
*Today we are excited to announce that DeepMind will start using #TensorFlow  for all future research, 
enabling the pursuit of ambitious goals at much larger scale and an even faster pace*


[DeepMind moves to TensorFlow]: http://googleresearch.blogspot.kr/2016/04/deepmind-moves-to-tensorflow.html

# Installation

### Requirements

* Python 2.7 or 3.3+
* GPU - CUDA Toolkit 7.5 and cuDNN v4

### Current Nvidia Card

현재 그래픽 카드 모델을 알고 싶을때는...
{% highlight bash %}
lspci -vnn | grep -i VGA -A 12
{% endhighlight %}

### CUDA Toolkit

Cuda Toolkit은 **/usr/local/cuda**안에 설치가 되어 있어야 합니다.<br>
[Download CUDA Toolkit][Download CUDA Toolkit]

다음의 Dependencies를 설치해줍니다.

{% highlight bash %}
sudo apt-get install libglu1-mesa libxi-dev libxmu-dev
{% endhighlight %}

CUDA Toolkit설치시 GPU Drive, CUDA, Nsight 등이 전부다 깔림니다.<br>
아래의 주소에서 RUN파일을 다운로드 받습니다.<br>
[https://developer.nvidia.com/cuda-downloads][cuda-toolkit]

1. 다운받은 폴더로 들어갑니다.
2. chmod로 실행파일로 바꿔줍니다.
3. CTRL + ALT + F3 또는 CTRL + ALT + F1 또는 Ubuntu Recovery Mode -> Command Shell
4. 로그인
5. init 3
6. sudo service lightdm stop
7. sudo su
8. ./NVIDIA*.run 파일 실행
9. reboot

만약 Unsupported Compiler 에러가 나면은, --override compiler 옵션을 붙여줍니다. 

{% highlight bash %}
$ cuda_7.5.18_linux.run --override compiler
{% endhighlight %}

그 다음으로 .bashrc에 다음을 추가해줍니다.

{% highlight bash %}

# CUDA & CUDNN
export PATH=$PATH:/usr/local/cuda/bin:/usr/local/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/lib
{% endhighlight %}

설치가 잘되었는지 확인은 다음과 같이 합니다.

{% highlight bash %}
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2015 NVIDIA Corporation
Built on Tue_Aug_11_14:27:32_CDT_2015
Cuda compilation tools, release 7.5, V7.5.17
{% endhighlight %}


### Reinstallation

{% highlight bash %}
sudo apt-get remove --purge nvidia*
sudo apt-get remove --purge  xserver-xorg-video-nouveau
{% endhighlight %}

### [Error] unsupported GNU version! gcc versions later than 5 are not supported!

4.x 버젼의 gcc를 쿠다가 설치된 곳으로 softlink걸어주면 해결됨<br>
먼저 GCC 4.8 그리고 G++ 4.8을 설치해줍니다. (두개의 버젼이 서로 동일해야합니다.)

{% highlight bash %}
sudo apt-get install python-software-properties
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-4.8
sudo apt-get install g++-4.8
 
sudo rm /usr/bin/g++
sudo ln -s /usr/bin/g++-4.8 /usr/bin/g++

sudo rm /usr/bin/gcc
sudo ln -s /usr/bin/gcc-4.8 /usr/bin/gcc
{% endhighlight %}

gcc-4.8을 쿠다가 설치된 곳으로 softlink를 걸어줍니다.

{% highlight bash %}
sudo ln -s /usr/bin/gcc-4.8 /usr/local/cuda/bin/gcc
{% endhighlight %}

### [Error] The driver installation is unable to locate the kernel source

CUDA 드라이버 설치시, Linux Kernel source가 필요한데, source가 안보여서 생기는 문제. 

[Download Nvidia Driver][Download Nvidia Driver]

{% highlight bash %}
sudo apt-get install linux-headers-`uname -r`
./NVIDIA-Linux-x86_64-367.18.run
{% endhighlight %}

또는 

{% highlight bash %}
sudo apt-get install linux-headers-generic
{% endhighlight %}

### CUDA + cuDNN

또한 cuDNN v4가 설치되어 있어야 합니다. v5설치시 TensorFlow는 source로 빌드되어야 합니다.<br>
[Download cuDNN][Download cuDNN]

*현재 TensorFlow가 cuDNN 4.0을 지원하고 있기 때문에 5.0대신에 4.0을 받습니다.<br>
다운을 받고 압축을 풀면 다음과 같은 구조로 되어 있습니다.

{% highlight bash %}
├── include
│   └── cudnn.h
└── lib64
    ├── libcudnn.so -> libcudnn.so.4
    ├── libcudnn.so.4 -> libcudnn.so.4.0.7
    ├── libcudnn.so.4.0.7
    └── libcudnn_static.a
{% endhighlight %}

안에 들어 있는 파일들을 CUDA Toolkit이 설치된 /usr/local/cuda 에 카피해주면됩니다.

{% highlight bash %}
tar xvzf cudnn-7.5-linux-x64-v4.tgz
chmod 644 cuda/include/*
sudo cp -r ./cuda/* /usr/local/cuda/
{% endhighlight %}


### Install Tensorflow from Source

TensorFlow build버젼은 기본적으로 CUDA Toolkit 7.5 그리고 cuDNN v5 와 작동하지만,<br> 
그 이상의 버젼에서 돌리기 위해서는 반드시 source에서 설치해야 합니다.

<bold style="color:red; font-weight:bold;">Pascal Architecture는 CUDA 8.0부터 지원됩니다. 즉 GTX1080, GTX1070 사용지 반드시 source로 install해야 합니다.</bold>

**Install Dependencies**

{% highlight bash %}
sudo apt-get install libcurl3-dev
{% endhighlight %}

**Bazel설치하기**

반드시 0.4 이상의 버젼 (0.3은 안됨)을 설치해야 합니다. 

{% highlight bash %}
$ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
$ curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
{% endhighlight %}

Update and Install Bazel

{% highlight bash %}
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
sudo apt-get update
sudo apt-get install bazel
sudo apt-get upgrade bazel
{% endhighlight %}

0.4 이상의 버젼인지 확인합니다.
{% highlight bash %}
$ bazel version
Build label: 0.4.3
{% endhighlight %}

**Install other dependencies**

{% highlight bash %}
# For Python 2.7:
$ sudo apt-get install python-numpy swig python-dev python-wheel
# For Python 3.x:
$ sudo apt-get install python3-numpy swig python3-dev python3-wheel
{% endhighlight %}

**Check CUDA Version**

{% highlight bash %}
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2015 NVIDIA Corporation
Built on Tue_Aug_11_14:27:32_CDT_2015
Cuda compilation tools, release 7.5, V7.5.17
{% endhighlight %}

**Install TensorFlow from source**

{% highlight bash %}
git clone https://github.com/tensorflow/tensorflow
{% endhighlight %}

{% highlight bash %}
cd tensorflow
./configure
{% endhighlight %}

configure시에 [Cuda Compute Capabilities][Cuda Compute Capabilities] 를 참고<br>
설치는 먼저 pip package를 만들고 그것을 설치합니다.

| GPU | Compute Capability |
|:----|:-------------------|
| GeForce GTX 1080	| 6.1  |
| GeForce GTX 1070	| 6.1  |
| GeForce GTX 980	| 5.2  |
| GeForce GTX 970	| 5.2  |
| GeForce GTX 960	| 5.2  |


{% highlight bash %}
$ bazel clean
$ bazel build --copt=-march=native --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ sudo pip install /tmp/tensorflow_pkg/tensorflow-*.whl
{% endhighlight %}

| Option | Description | me |
|:-------|:------------|:---|
| mavx  | Advanced Vector Extensions - [특정 프로세서](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX) 에서만 지원 (2008년 이후 Intel CPU에서 지원) | X |
| mavx2  | Advanced Vector Extensions 2 - [특정 프로세서](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2) 에서만 지원 (2008년 이후 Intel CPU에서 지원) | X |
| fma    | FMA instruction set - 2013년 이후 Intel CPU에서 지원 | X |
| -march=native | GCC Compilation 옵션 - 그냥 사용하면 됨 | | 

에러가 나면 bazel build시에 copt를 삭제 해보는것도 방법입니다.

{% highlight bash %}
$ bazel build --copt=-march=native --copt=-mfpmath=both -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
$ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ sudo pip install /tmp/tensorflow_pkg/tensorflow-*.whl
{% endhighlight %}


### 설치 확인 

설치 확인 방법은 ipython후 다음과 같이 합니다.

{% highlight python %}
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess =tf.Session() # 이때 GPU Device를 찾고 결과를 보여줍니다.
print(sess.run(hello)) # Hello, TensorFlow!

a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b)) # 42
{% endhighlight %}


# TensorFlow 101

<img src="{{ page.static }}tensorflow-logo.jpg" class="img-responsive img-rounded">

### Building Graph and Operations + Choosing GPU Device

그래프란 하나의 큰 흐름을 말하는 것이며 (Data Flow), 기본적으로 default graph가 자동으로 등록이 되며, 
그래프에는 여러개의 set of operations들을 nodes로 갖고 있을수 있습니다. 

Operations 들은 op 또는 ops로 쓸수 있으며, 하나의 수학적 공식이라고 생각하면 됩니다. 
즉 여러개의 수학적 공식들 (Operations)들이 모여서 하나의 그래프를 이루게 됩니다.
 
아래의 예제에서는 default graph가 3개의 nodes을 갖고 있습니다.
 
* 2개의 constants ops (matrix1, matrix2)
* 1개의 matmul() op 



{% highlight python %}
%pylab inline
import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[1,2,3], [4,5,6]])
matrix2 = tf.constant([[0, 1], [2, 2], [7, 9]])
product_matrix = tf.matmul(matrix1, matrix2)

session = tf.Session()
result = session.run(product_matrix)
session.close()
print result # [[25 32] [52 68]]
{% endhighlight %}

이때 GPU가 여러대일경우 선택해서 처리를 할 수 있습니다.

{% highlight python %}
with tf.device("/gpu:1"):
    result = session.run(product_matrix)
{% endhighlight %}


* **/cpu:0**: The CPU of your machine.
* **/gpu:0**: The GPU of your machine, if you have one.
* **/gpu:1**: The second GPU of your machine, etc.

### Launching the graph in a distributed session
  
먼저 TensorFlow Server를 각각의 Cluster machines들에서 띄워놓습니다.

{% highlight python %}
with tf.Session("grpc://example.org:2222") as sess:
  # Calls to sess.run(...) will be executed on the cluster.
  ...
{% endhighlight %}


### Variables

{% highlight python %}
# update -> state += one
state = tf.Variable(0, name='counter')
one = tf.constant(1)
update = tf.assign(state, tf.add(state, one))

# Launch the graph and run the ops.
with tf.Session() as sess:
    # Variables must be initialized by running an `init` Op after having
    # launched the graph.  We first have to add the `init` Op to the graph.
    # 세션(그래프)을 실행시킨이후에 반드시 init operation을 실행시켜야 합니다.
    # Variables 들은 모두 초기화가 되있어야 합니다.
    sess.run(tf.initialize_all_variables())
    
    # 현재 state의 값을 출력합니다. 즉 0 이 출력됩니다.
    print(sess.run(state)) 
    
    for i in range(3):
        sess.run(update)
        print(sess.run(state))
        
# output
0
1
2
3
{% endhighlight %}

데이터를 가져오기 위해서는 (Fetch), sess.run() 을 실행하면 됩니다.


### Feeds

tf.placeholder() 를 사용해서 sess.run()시에 어떤 arguments값으로 데이터를 전달 할 수 있습니다.

{% highlight python %}
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7., 2], input2:[2., 5.]}))

# output:
# [array([ 14.], dtype=float32)]
{% endhighlight %}


tf.placeholder(tf.float32, [2, 4]) 이렇게 2 dimensional arrays 로 만들었습니다.<br>
이경우 matrix는 다음과 같이 만들수 있습니다. [[1,2,3,4], [5,6,7,8]]

{% highlight python %}
d = tf.placeholder(tf.float32, [2, 4])
output = tf.transpose(d)
with tf.Session() as sess:
    feed = {d: [[1,2,3,4], [5,6,7,8]]}
    print sess.run([output], feed_dict=feed)
    
# [array([[ 1.,  5.],
#         [ 2.,  6.],
#         [ 3.,  7.],
#         [ 4.,  8.]], dtype=float32)]
{% endhighlight %}


### GPU 메모리 제한 두기

서버가 multi-user environment이라면, 반드시 allow_growth=True를 써줘서 메모리를 효율적으로 사용해야 합니다.

{% highlight python %}
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    pass
{% endhighlight %}

**TFLearn** 에서는 다음과 같이 합니다.

{% highlight python %}
tflearn.config.init_graph(gpu_memory_fraction=0.4, allow_growth=True)
{% endhighlight %}






[Download and Setup TensorFlow]: https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#download-and-setup
[Download CUDA Toolkit]: https://developer.nvidia.com/cuda-downloads
[Download cuDNN]: https://developer.nvidia.com/cudnn
[Download Nvidia Driver]: http://www.nvidia.com/Download/index.aspx?lang=en-kr
[Cuda Compute Capabilities]: https://developer.nvidia.com/cuda-gpus
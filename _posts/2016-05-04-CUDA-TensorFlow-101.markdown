---
layout: post
title:  "CUDA Toolkit, cuDNN, TensorFlow 101"
date:   2016-05-04 01:00:00
categories: "tensorflow"
static: /assets/posts/TensorFlow-101/
tags: ['CUDA', 'GTX960', 'Nvidia', 'Ubuntu', 'format']

---

<header>
<img src="{{ page.static }}tensorflow.jpg" class="img-responsive img-rounded img-fluid">
</header>

**2020년 1월 기준으로 업데이트 했습니다.**


# Install Nvidia Driver and CUDA for TensorFlow

## Ubuntu 18.04 NVIDIA Driver 

### Check devices

현재 그래픽 카드 모델을 알고 싶을때는...
{% highlight bash %}
lspci -vnn | grep -i VGA -A 12
{% endhighlight %}

어떤 드라이버를 설치해야 되는지는 다음의 명령어로 알수 있습니다. 

{% highlight bash %}
$ ubuntu-drivers devices
vendor   : NVIDIA Corporation
model    : GP104 [GeForce GTX 1070]
driver   : nvidia-driver-435 - distro non-free
driver   : nvidia-driver-410 - third-party free
driver   : nvidia-driver-415 - third-party free
driver   : nvidia-driver-430 - third-party free
driver   : nvidia-driver-440 - third-party free recommended
driver   : nvidia-driver-390 - third-party free
driver   : xserver-xorg-video-nouveau - distro free builtin
{% endhighlight %}

위에서 보면, `nvidia-driver-440`을 추천하고 있습니다. <br>
apt로 설치하면 됩니다. 

만약 위의 명령어를 실행할수 없지만, Nvidia graph card의 제품명을 알고 있다면 [Download NVIDIA Driver](https://www.nvidia.com/Download/driverResults.aspx/156086/en-us)
에 들어가서 **다운받지 말고** 버젼만 확인할수 있습니다. <br>
예를 들어 GTX-1070 의 경우 2020년 1월 25일 기준으로 440.44 가 가장 최신입니다. (다운받지 마세요. 우리는 apt로 해결할겁니다.)


### Install Dependencies

{% highlight bash %}
sudo apt-get install linux-headers-generic
sudo apt-get install libglu1-mesa libxi-dev libxmu-dev gcc build-essential
{% endhighlight %}


### Install Nvidia Driver

먼저 graphic-driver PPA를 추가합니다.

{% highlight bash %}
sudo add-apt-repository ppa:graphics-drivers
sudo apt update
{% endhighlight %}

아래의 명령어로 Nvidia driver를 자동으로 설치합니다.

{% highlight bash %}
sudo ubuntu-drivers autoinstall
{% endhighlight %}

만약 수동으로 설치를 하고자 한다면, 위에서 확인한 Nvidia driver version을 사용해서 설치를 할 수도 있습니다.<br>
(아래의 xxx부분을 확인된 버젼으로 변경해줘야 합니다. xxx그대로 사용하지 마세요. <br>
현재 글쓰는 시점에서 GTX-1070의 가장 최신 버젼은 440이네요. 즉 `sudo apt install nvidia-driver-440` )

{% highlight bash %}
sudo apt install nvidia-driver-xxx
{% endhighlight %}

### Disable Nouveau 

기존 우분투에서 지원하는 그래픽 드라이버를 제거합니다.<br>
Nvidia 그래픽 드라이버와 서로 충돌이 나면서 이후 문제가 생기는 것을 방지 합니다.

{% highlight bash %}
sudo vi  /etc/modprobe.d/nouveau-blacklist.conf 
{% endhighlight %}

`/etc/modprobe.d/nouveau-blacklist.conf` 에 아래의 내용을 넣습니다. 

{% highlight bash %}
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
{% endhighlight %}

이후 다음의 명령어로 부팅을 업데이트 해줍니다.

{% highlight bash %}
sudo update-initramfs -u
{% endhighlight %}

이후 reboot 시킵니다.

### Verify Nvidia Driver Installation

가장 쉬운 방법은 nvidia-smi로 체크하는 것입니다.

{% highlight bash %}
$ nvidia-smi

Fri Jan 24 23:20:34 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.48.02    Driver Version: 440.48.02    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1070    Off  | 00000000:01:00.0  On |                  N/A |
|  0%   56C    P8    14W / 230W |    834MiB /  8118MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1263      G   /usr/lib/xorg/Xorg                            40MiB |
|    0      1310      G   /usr/bin/gnome-shell                          51MiB |
|    0      1567      G   /usr/lib/xorg/Xorg                           374MiB |
|    0      1696      G   /usr/bin/gnome-shell                         189MiB |
|    0      2082      G   ...rson/apps/pycharm-2019.3.2/jbr/bin/java    14MiB |
|    0      2343      G   ...uest-channel-token=16905822469259145173   158MiB |
+-----------------------------------------------------------------------------+
{% endhighlight %}



## Install CUDA Toolkit & CUDA

### Install CUDA Toolkit

[Tensorflow GPU Support](https://www.tensorflow.org/install/gpu) 에 들어가서 먼저, TensorFlow 가 지원하는 CUDA버젼을 알아야 합니다.<br>
현재 2020년 1월의 기준으로 CUDA 10.1 을 요구하고 있습니다.  

[Download CUDA](https://developer.nvidia.com/cuda-downloads) 에서 맞는 CUDA 버젼을 다운로드 받습니다.
참고로 10.1의 경우 [legacy releases](https://developer.nvidia.com/cuda-10.1-download-archive-update2)에 들어가서 다운로드 받아야 합니다.

문서상에 따르면 10.1의 경우 다음과 같이 설치합니다. (2020년 1월 기준)<br>
이때 우리는 이미 `sudo apt install nvidia-driver-xxx` 명령어로 driver를 최신으로 설치한 상태이기 때문에, <br>
Drvier는 따로 설치할 필요 없습니다. 따라서 Nvidia drvier는 최신으로 사용하고, CUDA Toolkit은 TensorFlow에 맞춰서 <br>
legacy 를 사용하는 케이스가 생길수 있습니다.  

{% highlight bash %}
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sudo sh cuda_10.1.243_418.87.00_linux.run
{% endhighlight %}

이후 .bashrc에 다음을 설정합니다. 

{% highlight bash %}
# CUDA & CUDNN
export CUDAHOME=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin:/usr/local/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDAHOME/lib64:$CUDAHOME/lib:/usr/local/lib:$CUDAHOME/extras/CUPTI/lib64/
{% endhighlight %}


### Install cuDNN

[cuDNN 다운로드](https://developer.nvidia.com/cudnn) 에서 동일한 버젼의 (현재 문서에서는 10.1) cuDNN을 찾아서 다운로드 합니다.

다운로드 받은 cuDNN의 구조는 다음과 같습니다. 

{% highlight bash %}
cuda
├── include
│   └── cudnn.h
├── lib64
│   ├── libcudnn.so -> libcudnn.so.7
│   ├── libcudnn.so.7 -> libcudnn.so.7.6.5
│   ├── libcudnn.so.7.6.5
│   └── libcudnn_static.a
└── NVIDIA_SLA_cuDNN_Support.txt
{% endhighlight %}

{% highlight bash %}
tar xvzf cudnn-7.5-linux-x64-v4.tgz
chmod 644 cuda/include/*
sudo cp -P ./cuda/lib64/* /usr/local/cuda/lib64/
sudo cp ./cuda/include/cudnn.h /usr/local/cuda/include/
{% endhighlight %}



## TensorFlow

매우 쉽습니다.

{% highlight bash %}
sudo pip3 install tensorflow-gpu keras
{% endhighlight %}

설치 확인은 다음과 같이 합니다.

{% highlight python %}
import tensorflow as tf
tf.config.list_physical_devices('GPU')

#  [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
{% endhighlight %}








# TensorFlow 101

<img src="{{ page.static }}tensorflow-logo.jpg" class="img-responsive img-rounded img-fluid">

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

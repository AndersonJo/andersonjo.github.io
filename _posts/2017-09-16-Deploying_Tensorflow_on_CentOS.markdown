---
layout: post
title:  "TensorFlow Deployment in Samsung"
date:   2017-09-16 01:00:00
categories: "os"
asset_path: /assets/images/
tags: ['CentOS', 'samsung']

---

# 개요

최근 삼성에 "외주"나가서 TensorFlow및 CentOS설치 작업및, 다중 GPU사용 방법을 알려줄 일이 생겼었습니다.<br>
귀차너 귀찬너 하면서 귀찮은일도 모든 다 피가 되고 뼈가 된다 스스로 주문을 걸며 ㅎ;; 작업한 후기입니다.<br>

해당 페이지에서는 다음을 공유합니다.

1. 인터넷이 끊긴 CentOS 내부망 환경에서의 TensorFlow및 각종 라이브러리 deployment
2. TensorFlow를 이용한 multi GPU 간단한 사용 예제

> 삼성에는 많은 계열사가 있습니다.<br>
> 특정 계열사를 지칭하지 않았으며, 해당 회사의 민감한 내용은 내용은 전혀 없습니다. <br>
> 따라서 특정 회사에 종속되는 내용이 아닌, 일반적인 기술적 내용만을 담고 있습니다.


# CentOS Libraries Installation

환경은 CentOS 7.2이고, 내부망이기 때문에 모든 설치는 RPM으로 가져가서 설치작업을 마무리해야하는 상황입니다. <br>
따라서 모든 Linux & Python packages들은 모두 RPM 또는 Wheel 파일 형식으로 가져가서 설치해야합니다.

### Path 설정

일단 ~/.bashrc파일을 열어서 Path부터 지정합니다.<br>
anderson은 제 이름이니까.. 다른걸로 바꾸면 됩니다.

{% highlight bash %}
# CUDA & CUDNN
export PATH=$PATH:/usr/local/lib64:/usr/local/cuda/bin:/usr/local/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64:/usr/local/lib64/:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/lib

# added by Anaconda2 4.2.0 installer
export PATH="/home/anderson/anaconda2/bin:$PATH"

# added by Anaconda3 4.2.0 installer
export PATH="/home/anderson/anaconda3/bin:$PATH"
{% endhighlight %}


다음을 실행시킵니다.

{% highlight bash %}
sudo vi /etc/ld.so.conf
{% endhighlight %}

파일을 열고 다음을 집어넣습니다.

{% highlight bash %}
include ld.so.conf.d/*.conf
/usr/local/lib64/
/usr/local/lib64
/usr/local/lib/
/usr/local/lib
{% endhighlight %}

설정내용을 적용시킵니다.

{% highlight bash %}
sudo ldconfig -v
{% endhighlight %}

### Development Tools

앞서 말했듯이 인터넷이 연결안된 offline환경이기 때문에 먼저 인터넷이 되는 환경에서 RPM을 모두 다운받은 후, 파일들을 들어가서 설치해야 합니다.<br>
다음의 명령어는 RPM을 모두 다운 받고 설치를 합니다. (당연히 sudo rpm -ivh \* 할때는 설치할 서버에서 해야 되겠죠.)

{% highlight bash %}
sudo yum groupinstall --downloadonly --downloaddir='devtools' 'Development Tools'
cd devtools
sudo rpm -ivh *
{% endhighlight %}

### NCurses (New Curses)

텍스트 기반의 GUI같은 환경을 제공하는 라이브러리 입니다.

{% highlight bash %}
wget https://ftp.gnu.org/gnu/ncurses/ncurses-6.0.tar.gz
tar -zxvf ncurses-6.0.tar.gz
cd ncurses-6.0
./configure
make
sudo make install
{% endhighlight %}


### DKMS

{% highlight bash %}
wget http://dl.fedoraproject.org/pub/epel/7/x86_64/d/dkms-2.3-5.20170523git8c3065c.el7.noarch.rpm
sudo rpm -ivh dkms-2.3-5.20170523git8c3065c.el7.noarch.rpm
{% endhighlight %}

### Kernel Update

인터넷이 된다면 다음과 같이 하면 됩니다.

{% highlight bash %}
sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
{% endhighlight %}

하지만 오프라인 상황이기 때문에 특정 버젼의 kernel, kernel-devel, kernel-headers의 rpm을 구글에서 찾아서 다운 받습니다.

{% highlight bash %}
sudo rpm -ivh kernel-devel-3.10.0-693.el7.x86_64.rpm
sudo rpm -ivh kernel-headers-3.10.0-693.el7.x86_64.rpm
{% endhighlight %}


# CUDA Installation

먼저 ctrl + alt + f6 눌러서 tty로 나간후 gdm, kde 같은 데스크탑을 꺼줍니다.

{% highlight bash %}
sudo service gdm stop
{% endhighlight %}

CUDA Toolkit 을 설치합니다.
{% highlight bash %}
sudo sh cuda_*.run
{% endhighlight %}

CUDNN 을 설치합니다.

{% highlight bash %}
# CUDNN 5.1
tar -zxvf cudnn-8.0-linux-x64-v5.1.tgz
chmod 644 cuda/include/*
sudo cp -r ./cuda/* /usr/local/cuda/
{% endhighlight %}


# Python Libraries Installation

### Anaconda

{% highlight bash %}
wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
sh Anaconda3-4.2.0-Linux-x86_64.sh
{% endhighlight %}

Python 2.7은 virtual environment 를 만들어 줍니다.

{% highlight bash %}
conda create -n py27 python=2.7 anaconda
source activate py27
{% endhighlight %}


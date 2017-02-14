---
layout: post
title:  "Python 3 on Ubuntu"
date:   2016-09-22 01:00:00
categories: "aws"
asset_path: /assets/posts2/Language/
tags: ['format', 'Jupyter', 'Scipy', 'Numpy']

---

<header>
    <img src="{{ page.asset_path }}labtop.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small>제 노트북입니다.. ㅎㅎ 많은 스티커들 가운데.. Python 그리고 MSI GPU 스티커를 가장 좋아합니다.
    </small>
    </div>
</header>


# Python 3.6 + 

2017년 저물어가는 이때까지도.. 저는 Python2.7만 주구장창 사용해왔습니다. <br>
이번에 3.6의 기능들을 보고.. 이제는 3.x 버젼으로 갈아탈 시점이구나 생각이 들었습니다. <br>
Ubuntu에서 Python 3 를 설치하는 방법에 대해서 써놓겠습니다. <br>
물론 어렵지 않지만.. 항상 블로그에 남기는 이유는.. 급할때.. 항상 갖다 붙여넣기로 빠르게 일을 하기 위해서 ㅎㅎ

### Installing Python3.6 +

먼저 dependencies를 설치 합니다.

{% highlight bash %}
sudo apt-get install libreadline6-dev libbz2-dev libssl-dev libsqlite3-dev libncursesw5-dev libffi-dev libdb-dev libexpat1-dev zlib1g-dev liblzma-dev libgdbm-dev libffi-dev libmpdec-dev
{% endhighlight %}

[Python Download Page](https://www.python.org/downloads/)에서 3.x 버젼을 다운받고 다음과 같이 설치합니다.<br>
이때 make install 이 아니라 **make altinstall**을 해야 합니다.

{% highlight bash %}
tar xvf Python-*.tar.xz
cd Python-*
./configure
make
make test
sudo make altinstall 
{% endhighlight %}

Python3 명령어에 대한 soft link를 만들어줍니다.

{% highlight bash %}
sudo ln -s /usr/local/bin/python3.6 /usr/local/bin/python3
sudo ln -fs /usr/local/bin/python3.6 /usr/bin/python3
{% endhighlight %}

만약 no module named something 같은 에러가 나온다면.. python3 명령어를 쳤을때 source에 있는 python3를 실행시키는게 아니라.. 
Python3.5 (/usr/bin/python3 에 위치한)을 실행시키는 것일수도 있습니다. /usr/bin/python3 가 있다면.. 그냥 삭제. 

### Change system-wide Python

Ubuntu에서 파이썬을 실행시킬때 특정 버젼의  Python을 실행시키도록 변경할수 있습니다.<br>
다음은 Python alternatives를 모두 출력시킵니다.

{% highlight bash %}
$ sudo update-alternatives --list python
update-alternatives: error: no alternatives for python
{% endhighlight %}

Python 2.7을 기본으로 사용하게 하는 방법

{% highlight bash %}
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
{% endhighlight %}

Python  3.6을 기본으로 사용하게 하는 방법

{% highlight bash %}
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
{% endhighlight %}

--install 옵션은 여러개의 arguments를 받을수 있으면 symbolic links 를 만듭니다.<br>
마지막의 숫자는 priority로 생각하면 됩니다.

특정 Python 버젼을 선택하는 방법입니다.

{% highlight bash %}
update-alternatives --config python
{% endhighlight %}

### Installing PIP3 

[ensurepip - Bootstrapping the pip installer](https://docs.python.org/3/library/ensurepip.html)에서 자세한 내용을 참고 합니다.

{% highlight bash %}
sudo python3 -m ensurepip --upgrade
sudo ln -s /usr/local/bin/pip3.6 /usr/local/bin/pip3
{% endhighlight %}


pip version이 제대로 되어 있는지 확인합니다. (2.7 그리고 3.x가 서로 분리되어 있어야 합니다.)

{% highlight bash %}
$ pip --version
pip 9.0.1 from /usr/local/lib/python2.7/dist-packages (python 2.7)

$ pip3 --version
pip 9.0.1 from /usr/local/lib/python3.6/site-packages (python 3.6)
{% endhighlight %}

다음과 같은 에러가 나올 수 있습니다.<br>
**[Error] Could not find a version that satisfies the requirement wheel**

위와 같은 에러가 나올경우 먼저 wheel을 설치합니다.

[Wheel Git Repository](https://bitbucket.org/pypa/wheel)에서 소스를 다운받습니다.

{% highlight bash %}
sudo python3 setup.py install
{% endhighlight %}

python3를 실행한후 import wheel이 잘 되는지 확인해봅니다.

### Virtualenv on Python3 

{% highlight bash %}
virtualenv -p python3 test
source test/bin/activate
{% endhighlight %}


### Installing Libraries


**Data Analytics Tools**<br>
너무 느려서 -v 옵션을 꼭 붙이는게 좋습니다. (개 오래 걸림)

{% highlight bash %}
sudo pip3 install -v numpy scipy matplotlib ipython jupyter pandas sympy nose  
{% endhighlight %}

**Keras**

visualization을 위해서 다음을 설치합니다. (pydot-ng가 아니라 pydot설치시 keras visualization할때 에러를 일으킴)

{% highlight bash %}
sudo apt-get install graphviz
sudo pip3 install keras pydot-ng graphviz
{% endhighlight %}

**Ipython**

{% highlight bash %}
sudo pip3 install ipython[all]
sudo pip3 install ipywidgets
sudo jupyter nbextension enable --py --sys-prefix widgetsnbextension
{% endhighlight %}

**Jupyter**

{% highlight bash %}
python2 -m pip install ipykernel
python2 -m ipykernel install --user

python3 -m pip install ipykernel
python3 -m ipykernel install --user
{% endhighlight %}


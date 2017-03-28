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


### Installing PIP3 

[ensurepip - Bootstrapping the pip installer](https://docs.python.org/3/library/ensurepip.html)에서 자세한 내용을 참고 합니다.

{% highlight bash %}
sudo python3.6 -m ensurepip --upgrade
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
sudo python3.6 setup.py install
{% endhighlight %}

python3를 실행한후 import wheel이 잘 되는지 확인해봅니다.

### Virtualenv on Python3 

{% highlight bash %}
virtualenv -p python3.6 test
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
sudo pip3.6 install ipython[all]
sudo pip3.6 install ipywidgets
sudo jupyter nbextension enable --py --sys-prefix widgetsnbextension
{% endhighlight %}

**Jupyter**

{% highlight bash %}
python2 -m pip install ipykernel
python2 -m ipykernel install --user

python3.6 -m pip install ipykernel
python3.6 -m ipykernel install --user
{% endhighlight %}


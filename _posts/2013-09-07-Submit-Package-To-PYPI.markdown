---
layout: post
title:  "Submit Python Package to PyPI"
date:   2013-09-07 01:00:00
categories: "python"
asset_path: /assets/images/
tags: ['setup.py']

---

<header>
    <img src="{{ page.asset_path }}pypi_wallpaper.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;">
    <small><a href="https://unsplash.com/search/work?photo=DWui9DmfCXA">rawpixel.com의 사진</a>
    </small><br>
    현재 2017년 7월 19일. 문서 전체적으로 업데이트 했습니다. 많이 바꼈네요.
    </div>
</header>

# Configuration

### 먼저 할일

다음의 PyPI Package 웹싸이트에서 가입을 합니다.<br>
[https://pypi.org/](https://pypi.org/)

Requirements 를 설치 합니다.

{% highlight python %}
sudo pip3 install twine wheel
{% endhighlight %}



### setup.py

다음은 setup.py의 예제 입니다. <br>
실제 pewsql 이라는 제가 만든 Python package의 설정입니다.

{% highlight python %}
from setuptools import setup, find_packages

setup(
    name='pewsql',
    version='0.0.1',
    packages=['pewsql'],
    url='',
    license='',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    install_requires=['pony', 'pandas', 'numpy', 'psycopg2'],
    python_requires='>=3',
    author='anderson',
    author_email='a141890@gmail.com',
    description='Analytics Tools for RDBMS',
    keywords=['sql', 'analytics']
)
{% endhighlight %}

* **package**: 실제 package가 사용하는 디렉토리 이름과 동일해야 합니다.
* **url**: 보통 GIT 주소
* **license**: 개인적으로 MIT
* **classifiers**: [Classifiers 총 정리](https://pypi.python.org/pypi?%3Aaction=list_classifiers) 를 참고합니다. package를 분류하는데 사용됩니다.
* **install_requires**: Package를 사용하기 위해서 필요한 최소한의 dependencies.

### setup.cfg

PyPI에 REAME파일이 어디 있는지 알려줍니다.

{% highlight ini %}
[metadata]
description-file = README.md
{% endhighlight %}


### MANIFEST.in

{% highlight text %}
# Include the license file
include LICENSE.txt

# Include the data files
# recursive-include data *
{% endhighlight %}

### LICENSE.txt

다음은 MIT License 기본 형태 입니다.<br>
아래 <year> 그리고 <copyright holders> 를 바꿔주면 됩니다. (예: 2017 Anderson)

{% highlight text %}
Copyright (c) <year> <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
{% endhighlight %}


### .gitignore (optional)

추가적으로 다음과 같이  .gitignore 문서를 설정할 수 있습니다.

{% highlight text %}
# Project
.cache
.idea

# Package
dist
build
*.egg-info
MANIFEST

# Python
*.pyc
{% endhighlight %}




# Development Mode

프로젝트의 설치 그리고 테스트를 해볼 필요가 있습니다.<br>
다음의 명령어는 실제로 설치를 진행하며, install_requires에 있는 라이브러리들을 non-editable mode로 설치합니다.<br>
-e (editable) 옵션은 프로젝트는 편집할 수 있도록 합니다.

{% highlight python %}
sudo pip3 install -e .
{% endhighlight %}




# Packaging Your Project

### Source Distribution

아래의 명령어로 distribute한후, pip로 설치하면 build과정을 거치게 됩니다. (설치가 느려짐)

{% highlight python %}
python3.6 setup.py sdist
{% endhighlight %}


### Wheel

wheel로 설치시에는 sdist처럼 build과정을 거치지 않고 바로 설치가 됨으로 빠릅니다. 즉 built package.<br>
wheel을 사용시 몇가지 옵션이 있습니다.

| Mode | Description | Command |
|:-----|:------------|:--------|
| Universial Wheel | Pure Python으로 만들어졌으며, Python 2 그리고 3 모두 지원. | python setup.py bdist_wheel --universal |
| Pure Python Wheel | Pure Python으로 만들어졌지만, Python 2.x 또는 3.x 모두 지원하지는 않음 | python setup.py bdist_wheel |
| Platform Wheel | Compiled extensions등을 포함시 사용<br> bdist_wheel 명령어는 자동으로 Pure Python Code인지 감지함 | python setup.py bdist_wheel |

추가적으로 --universal 명령어 대신에 setup.cfg 파일안에 다음과 같이 설정할 수 있습니다.

{% highlight ini %}
[bdist_wheel]
universal=1
{% endhighlight %}


# Uploading Your Package to PyPI

### ~/.pypirc

**~/.pypirc** 파일을 생생한후 다음을 설정합니다.

{% highlight bash %}
vi ~/.pypirc
{% endhighlight %}

{% highlight ini %}
[pypi]
username = <username>
password = <password>
{% endhighlight %}

파일의 소유자만 읽기, 쓰기를 할 수 있도록 다음과 같이 권한을 변경합니다.

{% highlight bash %}
chmod 600 ~/.pypirc
{% endhighlight %}


### Uploading Your Distributions

{% highlight bash %}
twine upload dist/*
{% endhighlight %}

[https://pypi.org](https://pypi.org/) 에 들어가서 확인합니다.

### References

1. [https://github.com/pypa/sampleproject](https://github.com/pypa/sampleproject) 에서 예제를 볼 수 있습니다.
2. [https://packaging.python.org/tutorials/distributing-packages/#initial-files](https://packaging.python.org/tutorials/distributing-packages/#initial-files) 에서 Package 설치에 관한 공식 문서를 볼 수 있습니다.
---
layout: post
title:  "TensorFlow - SyntaxNet Tutorial"
date:   2016-09-16 01:00:00
categories: "tensorflow"
asset_path: /assets/posts2/TensorFlow/
tags: ['Python', 'NLU', 'NLP']

---

<header>
    <img src="{{ page.asset_path }}syntaxnet-long.png" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small>
       저에게는 꿈이 있습니다. 그리고 그 꿈은 TensorFlow덕분에 그 꿈을 해내는데 한 걸음 더 나아갈수 있을거 같습니다.<br>
    </small>
    </div>
</header>

# Installation

[SyntaxNet-Installation][SyntaxNet-Installation]을 참고

### Installing Dependencies on Ubuntu 

**Bazel**

[Bazel 0.3.1](https://github.com/bazelbuild/bazel/releases/tag/0.3.1)에 들어가서 Bazel Installer를 다운로드 받습니다. (*bazel-0.3.1-installer-linux-x86_64.sh* 다운)

{% highlight bash %}
$ chmod +x bazel-0.3.1-installer-linux-x86_64.sh
$ ./bazel-0.3.1-installer-linux-x86_64.sh --user
{% endhighlight %}

--user flag는 Bazel을 $HOME/bin 디렉토리에다가 설치를 합니다.<br>
다음을 추가합니다.

{% highlight bash %}
# Bazel
export PATH="$PATH:$HOME/bin"
{% endhighlight %}

버젼을 확인합니다.

{% highlight bash %}
$ bazel version
Build label: 0.3.1
{% endhighlight %}

**Swig**

{% highlight bash %}
$ sudo apt-get install swig
{% endhighlight %}

**Protocol buffers**

3.0.0b2 가 설치되어 있어야 합니다.

{% highlight bash %}
$ pip freeze | grep protobuf
protobuf==3.0.0b2 # 
{% endhighlight %}

3.0.0b2로 업그레이드 합니다. (버젼이 다르다면..)

{% highlight bash %}
$ pip install -U protobuf==3.0.0b2
{% endhighlight %}

**Asciitree** <small>console에서 parse tree를 그려줍니다.</small>

{% highlight bash %}
$ sudo pip install asciitree
{% endhighlight %}

**libcurl3-dev**

{% highlight bash %}
$ sudo apt-get install libcurl3-dev
{% endhighlight %}

**CUDA Version**

{% highlight bash %}
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2015 NVIDIA Corporation
Built on Tue_Aug_11_14:27:32_CDT_2015
Cuda compilation tools, release 7.5, V7.5.17
{% endhighlight %}

### NO STATUS 버그

[Undefined reference to symbol 'ceil@@GLIBC_2.2.5' Error][Undefined reference to symbol Error] Git Commit을 참고.

{% highlight bash %}
vi /syntaxnet/tensorflow/tensorflow/tools/proto_text/BUILD
{% endhighlight %}

다음을 추가시켜줍니다.

{% highlight bash %}
cc_library(
    name = "gen_proto_text_functions_lib",
    srcs = ["gen_proto_text_functions_lib.cc"],
    hdrs = ["gen_proto_text_functions_lib.h"],
    linkopts = [
        "-lm",
        "-lpthread",
    ] + select({
        "//tensorflow:darwin": [],
        "//conditions:default": ["-lrt"]
    }),    
    deps = [
        "//tensorflow/core:lib",
    ],
)
{% endhighlight %}

### Installing SyntaxNet

Build 그리고 Test 를 다음과 같이 합니다.

{% highlight bash %}
git clone --recursive --recurse-submodules https://github.com/tensorflow/models.git
cd models/syntaxnet/tensorflow
./configure
cd .. 
bazel clean
bazel test syntaxnet/... util/utf8/...
{% endhighlight %}

### Parsing from Standard Input

Parsey McParseface를 사용해서 shell에서 바로 테스트 해볼수 있습니다. <br>
주의깊게 볼 부분은 brought라는 동사가 문장의 root가 되었고, Bob이 주어, pizza가 목적어, to Alice가 전치사 부분으로 빠졌습니다.

{% highlight bash %}
$ echo 'Bob brought the pizza to Alice.' | syntaxnet/demo.sh
brought VBD ROOT
 +-- Bob NNP nsubj
 +-- pizza NN dobj
 |   +-- the DT det
 +-- to IN prep
 |   +-- Alice NNP pobj
 +-- . . punct
{% endhighlight %}

[SyntaxNet-Installation]: https://github.com/tensorflow/models/tree/master/syntaxnet#installation
[Bazel 0.2.2b]: https://github.com/bazelbuild/bazel/releases/tag/0.2.2b
[Undefined reference to symbol Error]: https://github.com/tensorflow/tensorflow/pull/3097/commits
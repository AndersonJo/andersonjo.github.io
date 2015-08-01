---
layout: post
title:  "GTX960 + Ubuntu 15.04 + Hello World"
date:   2015-08-03 02:00:00
categories: "ubuntu"
asset_path: /assets/posts/GTX960+Ubuntu+Cuda/
---

이번에 GTX960 그래픽카드를 질렀습니다.<br> 
설치 환경은 Ubuntu 15.04 + GTX960 인데, 혹시 나중에 다시 보게 될까봐 여기에다가 적습니다.

# ACPI PPC Probe failed

GTX960 디바이스를 읽지 못해서 생기는 에러입니다. 

그냥 warning 정도의 에러인데.. 이것과 상관없이.. 화면이 보이지 않는다면.. 

**nomodeset** 옵션을 주고 우분투를 설치또는 로그인하면 됩니다.
 
> 우분투 설치시에는 F6 (other options)를 눌러서 옵션을 지정할수 있습니다.


# Ubuntu 15.04 설치후..

### 32bit Libraries
GTX960 Driver를 설치하기전, 32bit 라이브러리를 설치해줍니다. <br>
해당 라이브러리는 또한 Android Studio사용시 설치 가능하게 해줍니다. 

{% highlight bash%}
sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32z1 lib32z1-dev
{% endhighlight %}


### Oracle Java
오라클 자바도 설치합니다. (이건 Pycharm, Nsight등의 IDE를 돌릴때 사용합니다) 

\* OpenJDK의 경우 성능이 좀 떨어지는 경우가 있습니다.

{% highlight bash%}
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java9-installer
{% endhighlight %}
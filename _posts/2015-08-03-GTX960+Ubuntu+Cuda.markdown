---
layout: post
title:  "GTX960 + Ubuntu 15.04 + Hello World"
date:   2015-08-03 02:00:00
categories: "ubuntu"
asset_path: /assets/posts/GTX960+Ubuntu15.04+Hello-World/

---

이번에 GTX960 그래픽카드를 질렀습니다.<br> 
설치 환경은 Ubuntu 15.04 + GTX960 인데, 혹시 나중에 다시 보게 될까봐 여기에다가 적습니다.

<img src="{{page.asset_path}}gtx960.jpg" class="img-responsive img-rounded">

### ACPI PPC Probe failed

GTX960 디바이스를 읽지 못해서 생기는 에러입니다. 

그냥 warning 정도의 에러인데.. 이것과 상관없이.. 화면이 보이지 않는다면.. 

**nomodeset** 옵션을 주고 우분투를 설치또는 로그인하면 됩니다.
 
> 우분투 설치시에는 F6 (other options)를 눌러서 옵션을 지정할수 있습니다.

### 32bit Libraries
GTX960 Driver를 설치하기전, 32bit 라이브러리를 설치해줍니다. <br>
해당 라이브러리는 또한 Android Studio사용시 설치 가능하게 해줍니다. 

{% highlight bash%}
sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32z1 lib32z1-dev
{% endhighlight %}


### Oracle Java
오라클 자바도 설치합니다. (이건 Nsight등의 IDE를 돌릴때 사용합니다) 

\* OpenJDK의 경우 성능이 좀 떨어지는 경우가 있습니다.

{% highlight bash%}
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java9-installer
{% endhighlight %}


### 벼루 (Optional)

{% highlight bash%}
sudo apt-get install uim uim-byeoru
uim-pref-gtk
{% endhighlight %}


### Command Prompt 설정 (Optional)

.bashrc 파일에 추가

{% highlight bash%}
parse_git_branch() {

    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'

}
export PS1='\[\033[00;36m\]\u:\[\033[0;33m\]\W$(parse_git_branch)>\[\033[00m\]'
{% endhighlight %}


### GTX960 Driver 

[http://www.geforce.com/drivers][gtx-driver]

위의 링크에서 드라이버를 다운받고 설치하면됨. 

### CUDA Toolkit

아래의 주소에서 RUN파일을 다운로드 받습니다.<br>
[https://developer.nvidia.com/cuda-downloads][cuda-toolkit]

1. 다운받은 폴더로 들어갑니다.
2. chmod로 실행파일로 바꿔줍니다.
3. CTRL + ALT + F3 
4. 로그인
5. init 3
6. sudo su
7. ./NVIDIA*.run 파일 실행
8. reboot

> reboot 이후에 로그인시 바로 로그 아웃이 되버리면, Ctrl + Alt + F3 누르고 home 에서 .Xauthority를 삭제시켜준다.



### CUDA Testing

Cuda샘플이 설치된 환경으로 이동한다면...

{% highlight bash%}

cd ./1_Utilities/deviceQuery
make
./deviceQuery

{% endhighlight %}


파일이 잘 실행이 되는지 확인을 합니다.

### Nsight

Toolkit 을 설치하게 되면 자동으로 Eclipse Nsight가 설치가 되어 있습니다.


{% highlight c%}

#include <stdio.h>

int main(){
    printf("Hello World\n");
    return 0;
}

{% endhighlight %}

위의 코드처럼 짠 다음에..

{% highlight bash%}

nvcc hello.c -o hello
hello

{% endhighlight %}

실행하면 뭐.. Hello World 프린트가 찍힙니다.


[gtx-driver]: http://www.geforce.com/drivers
[cuda-toolkit]: https://developer.nvidia.com/cuda-downloads

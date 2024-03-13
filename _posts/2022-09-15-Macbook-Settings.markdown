---
layout: post
title: "Macbook - Personal Settings"
date:  2022-09-15 01:00:00
categories: "format"
asset_path: /assets/images/
tags: ['brew', 'arm64', 'x86_64']
---


# 1. Basic Configuration


## 1.1 잠지기 방지 설정

노트북을 덮어도 꺼지지 않도록 설정하는 방법

```bash
# 잠자기 방지 설정 세팅
$ sudo pmset -c disablesleep 1

# 노트북 닫으면 정상적으로 꺼지도록 설정
$ sudo pmset -c disablesleep 0
```

## 1.1 Terminal & Brew & Bash

**Brew 설치**

{% highlight bash %}
# Brew 설치
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
$ echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.profile
$ eval "$(/opt/homebrew/bin/brew shellenv)"

# Bash 설치
$ brew install bash
{% endhighlight %}


**Bash로 변경**

{% highlight bash %}
$ chsh -s /bin/bash
{% endhighlight %}


`~/.bashrc` 또는 `~/.zshrc` 에 다음을 설정합니다.

{% highlight bash %}
export CLICOLOR=1
export LANG=en_US.UTF-8
{% endhighlight %}


**~/.bash_profile 에 작성**

{% highlight bash %}
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

# Hide default loging message
export BASH_SILENCE_DEPRECATION_WARNING=1

# Fancy colors in Bash
export CLICOLOR=1
export LSCOLORS=GxBxCxDxexegedabagaced

case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# Show Current Git Branch
parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}
export PS1='\[\033[00;36m\]\u:\[\033[0;33m\]\W$(parse_git_branch)>\[\033[00m\]'
{% endhighlight %}



## 1.2 키 바인딩 설정

**`HOME`, `END` 키를 줄의 처음, 끝으로 보내도록 설정**
 
{% highlight bash %}
$ mkdir -p ~/Library/KeyBindings/
$ vi ~/Library/KeyBindings/DefaultKeyBinding.dict
{% endhighlight %}

이후 다음의 내용을 넣습니다. 

{% highlight bash %}
{
    "\UF729"  = moveToBeginningOfLine:; // home
    "\UF72B"  = moveToEndOfLine:; // end
    "$\UF729" = moveToBeginningOfLineAndModifySelection:; // shift-home
    "$\UF72B" = moveToEndOfLineAndModifySelection:; // shift-end
    "@\U007F"  = deleteWordBackward:; // cmd + backspace
}
{% endhighlight %}

그외 대부분의 기본 세팅값들은 .. `/System/Library/Frameworks/AppKit.framework/Resources/StandardKeyBinding.dict` 에 있습니다.

**Terminal 에서의 변경**

`Preferences > Profiles > Settings > Keyboard.` 에서 변경이 가능합니다. 

 - 라인의 처음으로 이동: `\033OH`
 - 라인의 마지막으로 이동: `\033OF`


설정이 완료된 이후, 컴퓨터 재시작 필요.


## 1.3 그외 터미널 설정

**터미널 벨 끄기**

- Terminal -> Settings -> Advanced -> Bell
  - Visual Bell 체크 박스를 uncheck 으로 만듭니다. 



**Default Terminal as VIM**

`~/.bash_profile` 을 열고 다음을 추가 합니다. <br>

```bash
export EDITOR=/usr/bin/vim
```

**반복 키가 안눌리는 문제 해결**

```bash
$ defaults write -g ApplePressAndHoldEnabled -bool false
```

# 2. Python & Spark

> **매우 중요** <br>
> scipy는 M1 에서 Python 3.8 부터 지원이 됩니다. 
> Python 3.7 그 이하는 scipy 가 설치가 안됩니다. 


## 2.1 Preparation

이건 개인 설정이기 때문에, 상황에 따라 다를 수 있습니다. <br>
일단 **XCode**를 설치 합니다. 

```bash
brew install qt5 openblas gfortran

# Optional
brew install apache-arrow
brew install zmq
```



## 2.2 PIP Configuration for Security

pip 보안 설정을 해줍니다. 

```bash
mkdir ~/.pip
vi ~/.pip/pip.conf
```

아래와 같이 붙여 넣습니다. 

```yaml
[global]
trusted-host = pypi.python.org
               pypi.org
               files.pythonhosted.org
  
# Replace cert value 
# cert = /etc/ssl/cert.pem
cert = /etc/ssl/cert.pem
```


## 2.3 Install Python & Libraries

특히 M1 이라면 설치전 다음 명령어 실행이 필요합니다. 

```bash
# Python 설치 (원하는 버젼으로 수정)
pyenv install 3.10.13
pyenv global 3.10.13

# Numpy 설치시 필요한 명령어 입니다. (한번만 실행하면 되며, bashrc 같은 곳에 넣을 필요 없습니다.)
export OPENBLAS=$(brew --prefix openblas)
export CFLAGS="-falign-functions=8 ${CFLAGS}"

pip install --upgrade pip setuptools wheel
pip install -v numpy scikit-learn 
pip install --upgrade matplotlib ipython pandas pydot-ng graphviz 
pip install --upgrade jupyterlab mpl-interactions[jupyter] jupyterlab-code-formatter black isort autopep8
jupyter server extension enable --py jupyterlab_code_formatter
```

## 2.4 Jupyter Kernel 설정 

```bash
# 리스트 출력
$ jupyter kernelspec list

# 등록
$ python -m ipykernel install --user --name 3.10.13 --display-name "PyEnv 3.10.13"

# 삭제 
$ jupyter kernelspec remove <kernel_name> 
```


## 2.5 Spark

[https://spark.apache.org/downloads.html](https://spark.apache.org/downloads.html) 들어가서 최신 spark를 다운로드 받습니다. 

```bash
$ wget https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
$ tar -xvf spark-3.5.0-bin-hadoop3.tgz
$ mv spark-3.5.0-bin-hadoop3 ~/app/spark-3.5.0-bin-hadoop3

# pyspark 설치 
$ cd ~/app/spark-3.5.0-bin-hadoop3/python
$ python setup.py install 

# Log 설정
$ cd ~/app/spark-3.5.0-bin-hadoop3/conf
$ mv log4j2.properties.template  log4j2.properties
$ vi log4j2.properties 
``` 

아래와 같이 log4j2.properties 를 수정합니다. 

```bash
rootLogger.level = ERROR
```

## 2.6 Installing x86_64 brew package

M1, M2, M3 사용하면 arm64 아키텍쳐가 기본값으로 사용이 됩니다.<br> 
문제는 x86_64를 돌려야 할때가 있습니다. 이럴때는 x86_64 아키텍쳐 라이브러리를 설치하면 됩니다. 

```bash
$ arch -x86_64 /usr/local/bin/brew install <package name>
```

그냥 brew install 하게 되면 arm64 아키텍쳐 기반의 패키지가 설치가 됩니다. 

## 2.7 Docker Build for x86 architecture

Docker 빌드할때 X86 아키텍쳐 (인텔)로 빌드할수 있습니다.<br>
물론 할수는 있긴 있는데, 정말 한 10배는 느려지는 느낌입니다. 

그러니까 가급적 필요할때만 해야 합니다. 


```bash
$ docker build --platform linux/arm64 -t <tag-name>

# 또는 
$ docker buildx build --platform linux/arm64 -t <tag-name>
```


---
layout: post
title:  "Macbook - Personal Settings"
date:   2022-09-15 01:00:00
categories: "format"
asset_path: /assets/images/
tags: ['brew']
---


# 1. Basic Configuration

## 1.1 Terminal & Brew & Bash

**Brew 설치**

{% highlight bash %}
# Brew 설치
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
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
    "$\UF729" = moveToBeginningOfLineAndModifySelect햐ㅡion:; // shift-home
    "$\UF72B" = moveToEndOfLineAndModifySelection:; // shift-end
}
{% endhighlight %}

그외 대부분의 기본 세팅값들은 .. `/System/Library/Frameworks/AppKit.framework/Resources/StandardKeyBinding.dict` 에 있습니다.

**Terminal 에서의 변경**

`Preferences > Profiles > Settings > Keyboard.` 에서 변경이 가능합니다. 

 - 라인의 처음으로 이동: `\033OH`
 - 라인의 마지막으로 이동: `\033OF`


설정이 완료된 이후, 컴퓨터 재시작 필요.


## 1.3 그외 터미널 설정

- Terminal -> Settings -> Advanced -> Bell
  - Visual Bell 체크 박스를 uncheck 으로 만듭니다. 



## 1.3 Default Terminal as VIM

`~/.bash_profile` 을 열고 다음을 추가 합니다. <br>

```bash
export EDITOR=/usr/bin/vim
```
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

**Bash로 변경**

{% highlight bash %}
$ chsh -s /bin/bash
{% endhighlight %}


`~/.bashrc` 또는 `~/.zshrc` 에 다음을 설정합니다.

{% highlight bash %}
export CLICOLOR=1
{% endhighlight %}

**Brew 설치**

{% highlight bash %}
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
{% endhighlight %}



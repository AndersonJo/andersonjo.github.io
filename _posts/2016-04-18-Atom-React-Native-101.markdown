---
layout: post
title:  "Atom & React Native 101"
date:   2016-04-18 01:00:00
categories: "react"
static: /assets/posts/React-Native-101/
tags: ['node.js', 'jsx']

---

<img src="{{ page.static }}react-native-logo.jpg" class="img-responsive img-rounded">

# Get Started

### Requirements

* NPM 4.0 or newer
* IOS: Xcode 7.0 or higher
* Android: SDK

### JSX

React는 Javascript 대신에 Statically-typed language인 JSX를 사용합니다.

### Atom or Nuclide

Atom 설치이후에 apm을 통해서 react plugin 을 설치합니다.

{% highlight bash %}
apm install react
apm install atom-react-native-css
apm install goto
{% endhighlight %}

React를 전문적으로 개발하기 위해서는 Nuclide를 설치합니다.<br>

**keymap.cson** 에서 키맵을 재 설정합니다.

{% highlight bash %}
'atom-text-editor':
  'shift-space': 'autocomplete-plus:activate'
  'ctrl-shift-f': 'editor:auto-indent'
  'unset!': 'find-and-replace:find-next'

'.platform-linux atom-workspace atom-text-editor':
  'f3': 'goto:declaration'

'atom-workspace atom-text-editor:not([mini])':
  'alt-down': 'editor:move-line-down'
  'alt-up': 'editor:move-line-up'

'.platform-win32, .platform-linux':
  'ctrl-h': 'project-find:show'

'.platform-win32 atom-text-editor, .platform-linux atom-text-editor':
  'unset!': 'find-and-replace:find-next'
{% endhighlight %}


### Install

{% highlight bash %}
sudo npm install -g react-native-cli
{% endhighlight %}

### Start Project

{% highlight bash %}
react-native init ReactTutorial
cd ReactTutorial
react-native start
{% endhighlight %}

| Platform | How to run |
|:---------|:-----------|
| IOS | react-native run-ios | |
| Android | react-native run-android | 핸드폰 흔들고 -> Dev Settings ->  Host, Port를 변경 <br>(이때 http://는 붙이지 않습니다.)|

<img src="{{ page.static }}first_capture.png" class="img-responsive img-rounded">
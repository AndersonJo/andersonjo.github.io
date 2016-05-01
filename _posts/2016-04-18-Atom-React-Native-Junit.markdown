---
layout: post
title:  "Atom & React Native 101"
date:   2016-04-18 01:00:00
categories: "react"
static: /assets/posts/React-Native-101/
tags: ['node.js', 'jsx']

---

<img src="{{ page.static }}react-native-logo.jpg" class="img-responsive img-rounded">

# Atom or Nuclide

### Plugins

Atom 설치이후에 apm을 통해서 react plugin 을 설치합니다.

{% highlight bash %}
apm install react
apm install atom-react-native-css
apm install goto
apm install autocomplete-plus
{% endhighlight %}

React를 전문적으로 개발하기 위해서는 Nuclide를 설치합니다.<br>

### Key Maps

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



# Get Started

### Requirements

* NPM 4.0 or newer
* IOS: Xcode 7.0 or higher
* Android: SDK

*React는 Javascript 대신에 Statically-typed language인 JSX를 사용합니다.*

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


### Tutorial 

다음의 Rotten Tomato의 JSON데이터를 받아서 ListView에 올려놓는 튜토리얼을 볼 수 있습니다.

[react-tutorial-git-repository][react-tutorial-git-repository]

<img src="{{ page.static }}capture2.png" class="img-responsive img-rounded">


[react-tutorial-git-repository]: https://github.com/AndersonJo/react-tutorial


# JSX 101

### Installation

{% highlight bash %}
sudo npm install -g jsx
{% endhighlight %}


### Hello World

{% highlight javascript %}
class _Main {
    static function main(args : string[]) : void {
        log "Hello, world!";
        log args;
    }
}
{% endhighlight %}

{% highlight bash %}
$ jsx --run hello.jsx  asdf qwer 1234
Hello, world!
[ 'asdf', 'qwer', '1234' ]
{% endhighlight %}

* _Main클래스는 static function인 main을 갖고 있으며, 즉 class method입니다.
* main method는 args라는 array of strings를 받으며, 리턴값은 void입니다.
* log는 Javascript의 console.log와 매핑되어 있습니다.


{% highlight javascript %}
class _Main{
  static function main(args : string[]) : void {
    var dog = new Animal();
    log dog; // { name: '', age: 0 }
  }
}

class Animal {
  var name = '';
  var age = 0;

  function constructor(){}
  
  function constructor(name : string, age : number){
    this.set(name, age);
  }

  // function constructor(other : Animal){
  //   this.set(other);
  // }

  function set(name: string, age : number) : void{
    this.name = name;
    this.age = age;
  }

  function set(other : Animal) : void {
    this.set(other.name, other.age);
  }
}
{% endhighlight %}

* var name, var age의 경우 type을 안써놨는데, 이경우 초기 assignment되는 값의 타입에 의해 타입이 정해집니다.
* type parameters에 의한 overload가 존재합니다. (리턴값에 주의하자)

### Primitive Types

JSX에는 3개의 primitive types이 존재합니다.

| Name | Example | 
|:-----|:--------| 
| string | var a : string; |
| number | var b : number; |
| boolean | var c : boolean; |

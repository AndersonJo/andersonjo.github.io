---
layout: post
title:  "Unity3D on Ubuntu"
date:   2019-01-12 01:00:00
categories: "unity3d"
asset_path: /assets/images/
tags: ['Mono']
---


# Installation on Ubuntu

## Dependencies

먼저 dependencies 를 설치를 해줍니다.

{% highlight bash %}
sudo apt install libgtk2.0-0 libsoup2.4-1 libarchive13 libpng16-16 libgconf-2-4 lib32stdc++6 libcanberra-gtk-module
sudo apt-get install mono-devel
{% endhighlight %}


## Install Unity3D

아래 주소에서 Linux에서 배포된 Unity3D를 다운로드 받고 설치를 합니다.<br>
마지막 포스트를 찾고, Unity Hub for Linux를 다운로드 받습니다. 

원본 포스트는 아래의 링크를 참고 합니다.

* [https://forum.unity.com/threads/unity-on-linux-release-notes-and-known-issues.350256/](https://forum.unity.com/threads/unity-on-linux-release-notes-and-known-issues.350256/)

Unity Hub는 아래의 링크를 클릭 합니다.

* [https://forum.unity.com/threads/unity-hub-v-1-3-2-is-now-available.594139/](https://forum.unity.com/threads/unity-hub-v-1-3-2-is-now-available.594139/)


{% highlight bash %}
chmod +x UnityHubSetup.AppImage
{% endhighlight %}

executable로 만들어준후, **마우스로 더블 클릭**해서 실행합니다.

<img src="{{ page.asset_path }}unity3d-01.png" class="img-responsive img-rounded img-fluid">

License 선택시 personal로 합니다.  (Plus 또는 Pro 선택시 돈 나감..)

<img src="{{ page.asset_path }}unity3d-02.png" class="img-responsive img-rounded img-fluid">

설치되고 있는 화면의 모습

<img src="{{ page.asset_path }}unity3d-03.png" class="img-responsive img-rounded img-fluid">

설치 완료후 Unity Hub를 실행시킵니다.

아래의 이미지는 설치 완료후 Unity3D에서 제공하는 튜토리얼을 실행시킨 모습입니다.

<img src="{{ page.asset_path }}unity3d-04.png" class="img-responsive img-rounded img-fluid">
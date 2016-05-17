---
layout: post
title:  "Android Rooting"
date:   2016-05-10 01:00:00
categories: "android"
asset_path: /assets/posts/Android-Rooting/
tags: ['nmap', 'adb']

---

<div>
    <img src="{{ page.asset_path }}dynamite.jpg" class="img-responsive img-rounded">
</div>

<div style="color:red; font-weight:bold;">
아래의 모든 사항에 대한 모든 책임은 전적으로 실행한 당신에게 있습니다<br>
아래의 내용을 잘못 실행시 핸드폰에 피해가 생길수 있으며 이로 인한 2차 3차 모든 피해는 전적으로 당신에게 있습니다.<br>
또한 Rooting시 기록이 남게되고 이는 추후 A/S에서 피해가 생길수 있으므로 주의 요망<br>
Rooting후에 은행 관련 앱이용에 제한을 받을수도 있습니다.<br>
루팅된 디바이스는 해킹의 대상이 될 수도 있습니다.<br>
루팅과정중에 디바이스의 모든 데이터가 날라갈수도 또는 전원이 꺼지면 벽돌폰이 될 수도 있습니다.<br>

한마디로.. 다 니 책임!
</div>

# Android Rooting


먼저 디바이스의 port를 5555로 보도록 만듭니다. (tcpip는 adbd daemon을 restart 시킵니다.)

{% highlight bash %}
adb tcpip 5555
{% endhighlight %}

NMAP으로 확인해 봅니다. (192.168.0.16은 디바이스가 붙은 IP Address 입니다.)

{% highlight bash %}
$ nmap  192.168.0.16

Starting Nmap 6.47 ( http://nmap.org ) at 2016-05-12 11:20 KST
Nmap scan report for 192.168.0.16
Host is up (0.0079s latency).
Not shown: 999 closed ports
PORT     STATE SERVICE
5555/tcp open  freeciv

Nmap done: 1 IP address (1 host up) scanned in 0.24 seconds
{% endhighlight %}

디바이스에 연결을 합니다.

{% highlight bash %}
$ adb connect 192.168.0.16
connected to 192.168.0.16:5555

$ adb devices
List of devices attached
192.168.0.16:5555	device
LGF460K195764bf	device
{% endhighlight %}

디바이스를 reboot시켜줍니다. (reboot이후 다시 재접속이 필요 합니다.)

{% highlight bash %}
adb -s 192.168.0.16:5555 disable-verity
adb -s 192.168.0.16:5555 reboot
{% endhighlight %}
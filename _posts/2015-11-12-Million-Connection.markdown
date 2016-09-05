---
layout: post
title:  "1 million concurrent connections"
date:   2015-11-12 01:00:00
categories: "network"
asset_path: /assets/posts/Million-Connections/
tags: ['ulimit']
---
<header>
    <img src="{{ page.asset_path }}city.jpg" class="img-responsive img-rounded">
</header>

# Server Socket Descriptors

### Increasing Socket Descriptors

일반적으로 하나의 서버는 65535개의 connections을 받는다고 착각하는 분들이 꽤 있습니다. <br>
이유도 그럴듯한게.. 하나의 Listening IP Address에 여러개의 ports에 연결을 한다는... 
즉 ports는 2**16인 65536개가 있는데 이중 0번 포트는 제외한 65535개를 이용한다는.... 음..
뭐 하여튼 포인트는 65535개 이상으로 사용가능하고, 메모리 많다면 천만개까지도 늘릴수 있습니다. 
실제는 하나의 Listening IP Address와 하나의 Listening Port를 사용하지만, 각각의 클라이트마다 다른 Socket Descriptors를 사용하게 됩니다. 

먼저 hard 그리고 soft limit 을 확인하는 방법은 다음과 같이 합니다.<br>
soft 는 hard의 제한선을 넘을수가 없습니다.

{% highlight bash %}
ulimit -Hn
ulimit -Sn
{% endhighlight %}

Process당 socket descriptors의 최고치를 늘리는 것은 ulimit 명령어로 간단합니다.<br>
하지만 ulimit -n 100000 이렇게 하면.. 리부트시 다시 1024로 돌아옵니다.

{% highlight bash %}
sudo bash -c 'ulimit -n 1048576'
{% endhighlight %}

위의 명령어는 사용자별 open files 최고 수치를 변경하는 것입니다. <br>
여기서 soft와 hard가 구분지어지는데, 위의 명령어는 soft를 변경하는 것이고, soft는 hard의 제한선을 넘지 못합니다. 
hard 부분을 변경시키기 위해서는  /etc/security/limits.conf 를 변경해주어야 합니다.
변경 이후에는 re-login 또는 reboot가 필요합니다.

{% highlight bash %}
*               soft    nofile          10000000
*               hard    nofile          10000000
{% endhighlight %}

그다음으로 /etc/pam.d/common-session 열고, 다음을 추가시켜줍니다. <br>
session required pam_limits.so

{% highlight bash %}
sudo gedit /etc/pam.d/common-session
{% endhighlight %}

{% highlight bash %}
session required pam_limits.so
{% endhighlight %}

1048576은 ubuntu에서 기본적으로 제한해놓은 socket descriptors의 갯수입니다. (즉 2^20 == 1048576) 
그 이상으로 늘리기 위해서는 리눅스에서 걸어놓은 제한을 늘려야 합니다.

{% highlight bash %}
sudo su
echo 200005800 > /proc/sys/fs/nr_open
{% endhighlight %}

### Client Port Numbers

일단 Linux 서버에서 outgoing ports의 갯수를 확인해봅니다.
{% highlight bash %}
sysctl net.ipv4.ip_local_port_range
net.ipv4.ip_local_port_range = 32768	60999
{% endhighlight %}

즉 밖으로 나가는데 사용될수 있는 ephemeral ports는 32768~60999까지 사용되며 1~32767까지는 OS가 사용하게 됩니다.
사용가능한 ports의 갯수가 적기 때문에 굉장히 규모가 크거나, high bandwidth가 필요할때 이러한 튜닝을 하게 됩니다.

{% highlight bash %}
sudo sysctl -w net.ipv4.ip_local_port_range="500 65535"
{% endhighlight %}

또는

{% highlight bash %}
sudo echo 1024 65535 > /proc/sys/net/ipv4/ip_local_port_range
{% endhighlight %}

즉 이렇게 설정해 놓으면 
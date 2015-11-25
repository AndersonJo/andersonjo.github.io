---
layout: post
title:  "1 million concurrent connections"
date:   2015-11-12 01:00:00
categories: "network"
asset_path: /assets/posts/Million-Connections/
tags: ['']
---
<div>
    <img src="{{ page.asset_path }}city.jpg" class="img-responsive img-rounded">
</div>

# Server Socket Descriptors

### Increasing Socket Descriptors

일반적으로 하나의 서버는 65535개의 connections을 받는다고 착각하는 분들이 꽤 있습니다. <br>
이유도 그럴듯한게.. 하나의 Listening IP Address에 여러개의 ports에 연결을 한다는... 
즉 ports는 2**16인 65536개가 있는데 이중 0번 포트는 제외한 65535개를 이용한다는.... 음..
뭐 하여튼 포인트는 65535개 이상으로 사용가능하고, 메모리 많다면 천만개까지도 늘릴수 있습니다. 
실제는 하나의 Listening IP Address와 하나의 Listening Port를 사용하지만, 각각의 클라이트마다 다른 Socket Descriptors를 사용하게 됩니다. 

Process당 socket descriptors의 최고치를 늘리는 것은 ulimit 명령어로 간단합니다. 

{% highlight bash %}
sudo bash -c 'ulimit -n 1048576'
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
---
layout: post
title:  "Proxy Server with Squid"
date:   2020-10-10 01:00:00
categories: "ubuntu"
asset_path: /assets/images/
tags: ['squid']
---

# Proxy Server with Squid

remote 서버에서 다음과 같이 squid를 설치 그리고 설정을 합니다.

{% highlight bash %}
sudo apt-get update
sudo apt-get install squid

sudo cp  /etc/squid/squid.conf /etc/squid/original.conf
sudo vi /etc/squid/squid.conf
{% endhighlight %}

모두 삭제후 다음과 같이 작성합니다. 

{% highlight bash %}
http_port 3128
http_access allow all
acl all src all

# via off
forwarded_for off
# follow_x_forwarded_for deny all

shutdown_lifetime 1 seconds

cache_store_log none
cache_log /dev/null
cache deny all
{% endhighlight %}

설정후 재시작 합니다.

{% highlight bash %}
sudo service squid restart
sudo service squid status
{% endhighlight %}


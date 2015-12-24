---
layout: post
title:  "MySQL Semi Replication via Docker"
date:   2015-12-22 01:00:00
categories: "database"
asset_path: /assets/posts/MySQL-Replication/
tags: []
---
<div>
    <img src="{{ page.asset_path }}dolphins.jpg" class="img-responsive img-rounded">
</div>

MySQL 5.5부터 semisynchronous replication 을 plugin 형태로 지원을 합니다.<br> 
즉 master 그리고 slave모두 plugin이 깔려 있어야 합니다.
Semi Synchronous Replication은 재미있는게 일반적으로 Slave가 모두 적용을 다했는지 기다립니다.
하지만 timeout시에는 Asynchronous mode 로 전환이 되고, slave가 모두 따라잡으면 다시 Semi Synchronous mode로 변경이 됩니다.
또한 완전한 Synchronous Replication과 달리 Semi는 단 하나의 slave가 transaction을 받았다는 것을 reply하면 wait를 멈춤니다.
즉 모든 slave가 해당 transaction에 대해서 acknowledge 를 할때까지 기다리지 않습니다.
aknowledgement 의 의미는 slave 서버가 replay log에 적용하고 디스크에 flush했음을 의미합니다.
즉 실제로 데이터가 DB안에 들어갔음을 의미하지는 않습니다.

# MySQL Semi Replication via Docker

{% highlight bash %}
docker pull ubuntu:15.10
docker run -it ubuntu:15.10 bash
docker run --name mysql -e MYSQL_ROOT_PASSWORD=1234 -p 3350:3306 -d mysql:latest
{% endhighlight %}
 
# Loading SemiSync Replication Plugin

root 또는 super privilege 를 갖은 계정으로 DB에 접속합니다.

{% highlight sql %}
INSTALL PLUGIN rpl_semi_sync_master SONAME 'semisync_master.so';
{% endhighlight %}



쓰고있는중...
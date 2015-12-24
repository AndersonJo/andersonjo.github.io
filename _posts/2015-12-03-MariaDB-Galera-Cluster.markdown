---
layout: post
title:  "MariaDB Galera Cluster"
date:   2015-12-03 01:00:00
categories: "database"
asset_path: /assets/posts/MariaDB-Galera-Cluster/
tags: ["port forwarding"]
---
<div>
    <img src="{{ page.asset_path }}sea-lions-playing.jpg" class="img-responsive img-rounded">
</div>

# Intro

#### Features

* Synchronous Replication
* Multi Master topology
* 아무곳에나 read, write 실행 가능
* 자동 관리.. 즉 한 DB가 죽으면 자동으로 cluster에서 제외됨
* Automatic node joining
* True parallel replication, on row level

Galera Replication는 transaction commit을 할때 <br>
해당 transaction write set을 클러스터에 broadcasting 하고 적용하도록 합니다.

#### Installation

10.0 버젼이후부터 Galera Cluster 와 MariaDB가 함께 붙어서 제공되고 있습니다.<br>
하지만 그 이전버젼은 Galera Cluster를 따로 설치해야 합니다.

{% highlight bash %}
sudo apt-get install mariadb-server
{% endhighlight %}

#### Dockerinzing MariaDB

먼저 Network를 만듭니다.

{% highlight bash %}
docker network create mynetwork
docker network ls

NETWORK ID          NAME                DRIVER
5ad5a14327d4        mynetwork           bridge              
89c3406dab1e        bridge              bridge              
b6ce7303a798        none                null                
33ae8e65897a        host                host
{% endhighlight %} 

docker를 실행할때 --net=<네트워크 이름> 을 통해서 어디 네트워크를 사용할지 결정할수 있습니다.<br>
기본적으로 --net 옵션을 주지 않는다면 bridge (docker0 네트워크) 라는 network를 기본적으로 사용하게 됩니다.




{% highlight bash %}

docker run -d --name fission --net=host -p 0.0.0.0:3370:3370 cluster

docker pull mariadb
docker run -p 3306:3306 --name db01 -e MYSQL_ROOT_PASSWORD=1234 -d mariadb:10.0.22 --wsrep-new-cluster
{% endhighlight %}



--wsrep-new-cluster 의 의미는 연결할수 있는 cluster가 없고, 새로운 history UUID를 만듭니다.<br>
restarting server를 하면 새로운 UUID가 만들어지며, old cluster에 reconnect하지 않습니다.


[https://github.com/DominicBoettger/docker-mariadb-galera]: https://github.com/DominicBoettger/docker-mariadb-galera
[https://github.com/dockerfile/mariadb/blob/master/Dockerfile]: https://github.com/dockerfile/mariadb/blob/master/Dockerfile
[https://github.com/docker-library/mariadb/blob/034c283be05caa5e465047ce19f1770647eadd74/10.0/Dockerfile]: https://github.com/docker-library/mariadb/blob/034c283be05caa5e465047ce19f1770647eadd74/10.0/Dockerfile
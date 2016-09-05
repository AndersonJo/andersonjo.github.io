---
layout: post
title:  "MariaDB Galera Cluster via Docker"
date:   2015-12-03 01:00:00
categories: "database"
asset_path: /assets/posts/MariaDB-Galera-Cluster/
tags: ["docker", "wsrep"]
---
<header>
    <img src="{{ page.asset_path }}sea-lions-playing.jpg" class="img-responsive img-rounded">
</header>

# Intro

#### Features

* Synchronous Replication
* Multi Master topology
* 아무곳에나 read, write 실행 가능
* 자동 관리.. 즉 한 DB가 죽으면 자동으로 cluster에서 제외됨
* Automatic node joining
* True parallel replication, on row level


# Install & Configure Galera Cluster

10.1 버젼부터 Galera Cluster가 함께 제공되고 있습니다.<br>
[https://downloads.mariadb.org/mariadb/repositories/][mariadb-install-page] 여기에 들어가면, <br>
10.1을 깔수 있으며, 그냥 시키는대로 하면 됩니다.

#### Setting Firewall

| Port | 사용되는 곳 |
|:-----|:----------|
|3306| For MySQL client connections and State Snapshot Transfer that use the mysqldump method.|
|4567| For Galera Cluster replication traffic, multicast replication uses both UDP transport and TCP on this port.|
|4568| For Incremental State Transfer.|
|4444| For all other State Snapshot Transfer.|


#### my.cnf

{% highlight bash %}
[mysqld]
collation-server = utf8_unicode_ci
init-connect='SET NAMES utf8'
character-set-server = utf8

skip-host-cache
skip-name-resolve
#
# * Basic Settings
#
user            = mysql
pid-file        = /var/run/mysqld/mysqld.pid
socket          = /var/run/mysqld/mysqld.sock
port            = 3306
basedir         = /usr
datadir         = /var/lib/mysql
tmpdir          = /tmp
lc_messages_dir = /usr/share/mysql
lc_messages     = en_US
skip-external-locking

# Galera Cluster Required
wsrep_on=ON
wsrep_causal_reads=ON
binlog_format = ROW
innodb_autoinc_lock_mode=2
innodb_doublewrite=1
innodb_flush_log_at_trx_commit=0
wsrep_retry_autocommit=10

# Galera Provider Configuration
wsrep_provider=/usr/lib/galera/libgalera_smm.so
#wsrep_provider_options="gcache.size=32G"

# Galera Cluster Configuration
wsrep_cluster_name="test_cluster"
#wsrep_cluster_address="gcomm://first_ip,second_ip,third_ip"

# Galera Synchronization Congifuration
wsrep_sst_method=rsync
#wsrep_sst_auth=user:pass

# Galera Node Configuration
wsrep_node_address="node_ip"
wsrep_node_name="node_name"
{% endhighlight %}

#### Primary Node

중요한점은 Primary Node는 wsrep_cluster_address부분을 아무것도 설정하지 않습니다.

{% highlight bash %}
wsrep_cluster_address="gcomm://"
wsrep_node_address="primary_node_ip"
{% endhighlight %}

##### Additional Cluster Node




# Running Primary Node

Cluster의 첫번째 Node는 다음과 같이 실행시킵니다.

{% highlight bash %}
sudo service mysql start --wsrep-new-cluster
sudo service mysql restart --wsrep_new_cluster
{% endhighlight %}

상태체크는 다음과 같이 합니다.

{% highlight bash %}
systemctl status mariadb.service
{% endhighlight %}

클러스터 상태체크

{% highlight bash %}
SHOW STATUS LIKE 'wsrep_%';
{% endhighlight %}

wsrep_cluster_address 그리고 wsrep_node_address 이 부분이 중요한데,<br>
3306이 포트를 찾는게 아니라 4567 포트로 먼저 통신을 하게 됩니다. <br>

{% highlight bash %}
#wsrep_cluster_address="gcomm://first_ip:4567,second_ip:4567,third_ip:4567"<br>
wsrep_node_address="ip:4567"
{% endhighlight %}







# Dockerinzing MariaDB

#### Install Dockerized MariaDB 10.1

{% highlight bash %}
git clone git@github.com:AndersonJo/mariadb10.1-docker.git

docker build -t mariadb .
docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=1234 --name cluster1 -d mariadb
{% endhighlight %}


#### Connect to each other

Docker의 --link 는 Bidirectional Connection (즉 서로 연결되는 Connection) 을 지원하지 않습니다.<br>
(아직 Running도 안하는 Container에 연결을 시킬순 없겠죠)<br>
대신에 서로의 IP 주소로 들어갈수 있습니다.<br>
inspect로 현재 컨테이너의 IP Address를 알아낼수 있습니다.

{% highlight bash %}
docker inspect cluster1  | grep IPAddress

# 만약 Container 안이라면..
cat /etc/hosts
{% endhighlight %}

#### Create a primary node

{% highlight bash %}
docker run --name cluster01 -p 3306:3306 -e MYSQL_ROOT_PASSWORD=1234 -d mariadb mysqld --wsrep_new_cluster
{% endhighlight %}

MySQL에 Cilent로 로그인 한뒤, 다음의 명령어로 status를 확인할수 있습니다.

{% highlight bash %}
show status like 'wsrep%';
{% endhighlight %}

--wsrep-new-cluster 의 의미는 연결할수 있는 cluster가 없고, 새로운 history UUID를 만듭니다.<br>
restarting server를 하면 새로운 UUID가 만들어지며, old cluster에 reconnect하지 않습니다.


Primary Node 의 my.cnf 에 반드시 들어가야 할 내용

{% highlight bash %}
wsrep_cluster_address="gcomm://172.17.0.3"
wsrep_node_address="172.17.0.2"
wsrep_node_name="cluster01"
{% endhighlight %}

wsrep_cluster_address 에는 다른 nodes 의 주소를 , comma 를 통해서 써넣습니다.<br>
wsrep_node_address 에는 자신의 주소를 적어넣습니다.



#### Creating Network (Optional)

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


# 유용한 링크

* [https://mariadb.com/kb/en/mariadb/galera-cluster-system-variables/][https://mariadb.com/kb/en/mariadb/galera-cluster-system-variables/]


{% highlight bash %}
docker run -d --name fission --net=host -p 0.0.0.0:3370:3370 cluster

docker pull mariadb
docker run -p 3306:3306 --name db01 -e MYSQL_ROOT_PASSWORD=1234 -d mariadb:10.0.22 --wsrep-new-cluster
{% endhighlight %}

{% highlight bash %}
docker run -p 3306:3306 -p 4444:4444 -p 4567-4568:4567-4568 -e MYSQL_ROOT_PASSWORD=1234 -v /home/ubuntu/db/mysql:/var/lib/mysql -v /home/ubuntu/db/log/mysql:/var/log/mysql/ --name cluster -d mariadb mysqld --wsrep_new_cluster
{% endhighlight %}


[https://github.com/DominicBoettger/docker-mariadb-galera]: https://github.com/DominicBoettger/docker-mariadb-galera
[https://github.com/dockerfile/mariadb/blob/master/Dockerfile]: https://github.com/dockerfile/mariadb/blob/master/Dockerfile
[https://github.com/docker-library/mariadb/blob/034c283be05caa5e465047ce19f1770647eadd74/10.0/Dockerfile]: https://github.com/docker-library/mariadb/blob/034c283be05caa5e465047ce19f1770647eadd74/10.0/Dockerfile
[mariadb-install-page]: https://downloads.mariadb.org/mariadb/repositories/


[https://mariadb.com/kb/en/mariadb/galera-cluster-system-variables/]: https://mariadb.com/kb/en/mariadb/galera-cluster-system-variables/
---
layout: post
title:  "MariaDB Replication via Docker"
date:   2015-12-04 01:00:00
categories: "database"
asset_path: /assets/posts/MariaDB-Replication/
tags: ["port forwarding"]
---
<header>
    <img src="{{ page.asset_path }}sealions.jpg" class="img-responsive img-rounded img-fluid">
</header>

# Dockerinzing MariaDB
 
#### Installing MariaDB 10.0.22

[Dockerfile 10.0][https://github.com/docker-library/mariadb/blob/034c283be05caa5e465047ce19f1770647eadd74/10.0/Dockerfile]

해당 git을 clone하고 10.0 폴더 안에서 docker build시키면 됩니다. <br>
참고로 docker pull mariadb 해서 이미지 받으면 5.5 입니다. 

{% highlight bash %}
cd mariadb/10.0
docker build -t mariadb .
{% endhighlight %}


# Configuring Master

{% highlight bash %}
docker run -p 3306:3306 --name mariadb -e MYSQL_ROOT_PASSWORD=1234 -d mariadb
{% endhighlight %}


{% highlight bash %}
[client]
default-character-set=utf8
port            = 3306

[mysqld]
collation-server = utf8_unicode_ci
init-connect='SET NAMES utf8'
character-set-server = utf8

server-id               = 1
log_bin                 = /var/log/mysql/mariadb-bin
log_bin_index           = /var/log/mysql/mariadb-bin.index
#sync_binlog            = 1
expire_logs_days        = 10
max_binlog_size         = 100M
# slaves
#relay_log              = /var/log/mysql/relay-bin
#relay_log_index        = /var/log/mysql/relay-bin.index
#relay_log_info_file    = /var/log/mysql/relay-bin.info
#log_slave_updates
{% endhighlight %}

위에 있는 내용들을 찾아서 주석을 제거해주면 됩니다.

{% highlight bash %}
docker restart mariadb
{% endhighlight %}


SLAVE 권한을 MASTER에서 줍니다.<br>
그리고 MASTER에 대한 정보를 얻습니다.

{% highlight sql %}
CREATE USER `repl`@`%` IDENTIFIED BY '1234';
GRANT REPLICATION SLAVE ON *.* to `repl`;

FLUSH TABLES WITH READ LOCK;
SHOW MASTER STATUS;
+--------------------+----------+--------------+------------------+
| File               | Position | Binlog_Do_DB | Binlog_Ignore_DB |
+--------------------+----------+--------------+------------------+
| mariadb-bin.000001 |      599 |              |                  |
+--------------------+----------+--------------+------------------+

UNLOCK TABLES;
{% endhighlight %}

만약 MASTER DB가 사용중이었다면, mysqldump를 사용해서 이전의 데이터를 slave에 구성해야 합니다.

# Configuring Slave

{% highlight bash %}
docker run -p 3307:3306 --name slave --link mariadb:master --volume /home/ubuntu/data/mariadb/:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=1234 -d mariadb
docker exec -it slave bash
{% endhighlight %}

SLAVE의 my.cnf 설정을 다음과 같이 합니다.<br>
server-id는 master 와 달라야 합니다.

{% highlight bash %}
server-id               = 2
#report_host            = master1
#auto_increment_increment = 2
#auto_increment_offset  = 1
log_bin                 = /var/log/mysql/mariadb-bin
log_bin_index           = /var/log/mysql/mariadb-bin.index
# not fab for performance, but safer
#sync_binlog            = 1
expire_logs_days        = 10
max_binlog_size         = 100M
# slaves
relay_log               = /var/log/mysql/relay-bin
relay_log_index = /var/log/mysql/relay-bin.index
relay_log_info_file     = /var/log/mysql/relay-bin.info
replicate-do-db=www
log_slave_updates
{% endhighlight %}

다음과 같이 master의 DB또는 특정 table을 replicate할 수 있습니다.<br>
여러개가 있으면 여러번 쓰면 됩니다.

* replicate-do-db = db_name
* replicate-do-table = db_name.table_name

만약 전체 테이블에 대한 Replication을 구축하고 싶다면 다음과 같이 합니다.
{% highlight bash %}
binlog-ignore-db=information_schema
binlog-ignore-db=mysql

replicate-ignore-db=information_schema
replicate-ignore-db=mysql
{% endhighlight %}

Restart시켜줍니다.

{% highlight bash %}
docker restart slave
{% endhighlight %}

다음으로 SLAVE DB로 접속해서 설정해주면 됩니다.
 
{% highlight sql %} 
MariaDB [(none)]> CHANGE MASTER TO
  MASTER_HOST='master',
  MASTER_USER='repl',
  MASTER_PASSWORD='1234',
  MASTER_PORT=3306,
  MASTER_LOG_FILE='mariadb-bin.000001',
  MASTER_LOG_POS=0,
  MASTER_CONNECT_RETRY=10;
{% endhighlight %}


#### Master 에서 에러날시 

Master에서 에러가 나면  slave에서는 에러가 난 부분부터 동기화를 중단합니다.<br>
에러부분을 자동으로 skip 하기 위해서는 slave db -> my.cnf -> [mysqld] 안에다가 다음을 추가합니다.

{% highlight sql %} 
slave-skip-errors=all
{% endhighlight %}


[https://github.com/docker-library/mariadb/blob/034c283be05caa5e465047ce19f1770647eadd74/10.0/Dockerfile]: https://github.com/docker-library/mariadb/blob/034c283be05caa5e465047ce19f1770647eadd74/10.0/Dockerfile
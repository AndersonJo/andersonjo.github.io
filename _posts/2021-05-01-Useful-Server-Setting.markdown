---
layout: post
title:  "Boilerplate Configuration for Server"
date:   2021-05-01 01:00:00
categories: "server"
asset_path: /assets/images/
tags: ['mariadb', 'mysql', 'db']
---

그냥 매번하는 서버 설정을 남겨둠.  

# 1. Database 
## 1.1 MariaDB

`sudo vi /etc/mysql/mariadb.conf.d/50-server.cnf` 서버 설정을 합니다.<br>
`[mysqld]` 는 mysql server 그리고 mariadb server 둘다 적용이 됩니다. 

{% highlight bash %}
[mysqld]
bind-address = 0.0.0.0
character-set-server  = utf8
collation-server      = utf8_general_ci
init-connect='SET NAMES utf8'
{% endhighlight %}

`sudo vi /etc/mysql/mariadb.conf.d/50-client.cnf` 

{% highlight bash %}
[client]
default-character-set = utf8
{% endhighlight %}


MariaDB 리스타트 후에 Root권한으로 접속. 

{% highlight bash %}
$ sudo service mysqld restart
$ sudo mariadb
{% endhighlight %}

접속후에 character_set 확인을 합니다.

{% highlight bash %}
SHOW variables like 'character_set%';

+--------------------------+----------------------------+
| Variable_name            | Value                      |
+--------------------------+----------------------------+
| character_set_client     | utf8mb4                    |
| character_set_connection | utf8mb4                    |
| character_set_database   | utf8mb4                    |
| character_set_filesystem | binary                     |
| character_set_results    | utf8mb4                    |
| character_set_server     | utf8mb4                    |
| character_set_system     | utf8                       |
| character_sets_dir       | /usr/share/mysql/charsets/ |
+--------------------------+----------------------------+
{% endhighlight %}

Port 설정이 잘 되었는지 확인합니다.


{% highlight bash %}
# LISTEN 하고 있는 port를 체크 합니다. 
netstat -tlnp

# port 열려 있는지 확인
# nmap [서버주소]
{% endhighlight %}
---
layout: post
title:  "Hortonworks Hadoop"
date:   2015-09-08 01:00:00
categories: "hadoop"
asset_path: /assets/posts/Hortonworks-Hadoop/
tags: ['HDP']

---
<div>
    <img src="{{ page.asset_path }}HWX-RGB-full-tagline.png" class="img-responsive img-rounded">
</div>

Ubuntu 14.10 (trusty version) 에 Hortonworks Hadoop with Ambari를 설치하는 방법입니다.

# Hortonworks Hadoop with Ambari

### Requirements

* Oracle Java 1.7 or 1.8
* Metastore Database (Postgres 8.x or 9.3+, Mysql 5.6, oracle 11g r2)
  
데이터베이스에는 다음의 유저들이 필요합니다. <br>
(참고로 Hive default metastore는 derby이지만, production에서는 Derby가 사용되지 않습니다. - not suppported)<br>
그냥 mysql-server설치 하면 됩니다.

{% highlight bash %}
sudo apt-get install libmysql-java ntp
{% endhighlight %}


### Configuring CentOS

{% highlight bash %}
sudo yum install mlocate
{% endhighlight %}

### Default Ports

| Service | Name | Port | Protocol |
|:--------|:-----|:-----|:---------|
| Ambari | Ambari Web Interface | 8080 | HTTP |
| Ambari | Handshake Port for Ambari Agents to Ambari Server | 8440 | HTTPS |
| Ambari | Registration & Heartbeat Port <br>for Ambari Agents to Ambari Server | 8441 | HTTPS |
| HDFS | mapreduce.jobhistory.address | 10020 | IPC |
| HDFS | mapreduce.jobhistory.admin.address | 10033 | IPC |
| HDFS | mapreduce.jobhistory.webapp.address | 19888 | WEB |
| NameNode | NameNode | 50070 | |
| NameNode | NameNode RPC | 8020 | | 
| Secondary NameNode | HDFS Secondary NameNode | 50090 | | 
| ZooKeeper | ZooKeeper Client | 2181 | |



### Disable Transparent Huge Pages (Optional)

Hadoop의 Performance의 영향을 줄 수 있는 부분입니다.<br>
일단 확인하는 방법은 다음과 같습니다.

{% highlight bash %}
cat /sys/kernel/mm/transparent_hugepage/enabled
{% endhighlight %}

| [always] | truned on |
| [never] | turned off |

disable해주기 위해서는 다음의 파일을 엽니다. 

{% highlight bash %}
sudo vi /etc/default/grub
{% endhighlight %}

다음의 예제처럼 **transparent_hugepage=never**를 추가시켜줍니다.

{% highlight bash %}
GRUB_CMDLINE_LINUX_DEFAULT="console=tty1 console=ttyS0 transparent_hugepage=never"
{% endhighlight %}

{% highlight bash %}
sudo update-grub
{% endhighlight %}

### Installing Ambari

[hortonworks hadoop with ambari][hortonworks hadoop with ambari] 링크를 참조

{% highlight bash %}
$ sudo wget -nv http://public-repo-1.hortonworks.com/ambari/ubuntu14/2.x/updates/2.2.2.0/ambari.list -O /etc/apt/sources.list.d/ambari.list

$ sudo apt-key adv --recv-keys --keyserver keyserver.ubuntu.com B9733A7A07513CAD
$ sudo apt-get update
$ sudo apt-get install ambari-server
$ sudo apt-get install ambari-agent
$ sudo apt-get install ambari-metrics-assembly
{% endhighlight %}

설치후 제대로 모두 설치됐는지 다음과 같이 확인합니다.

{% highlight bash %}
apt-cache showpkg ambari-server
apt-cache showpkg ambari-agent
apt-cache showpkg ambari-metrics-assembly
{% endhighlight %}



만약 ambari-server start를 했을때 ImportError: No module named ambari_commons.exceptions이 발생한다면 다음과 같이 합니다.

{% highlight bash %}
sudo cp -r /usr/lib/python2.6/site-packages/* /usr/local/lib/python2.7/dist-packages/
{% endhighlight %}






### Mysql Setting

{% highlight sql %}
CREATE USER `ambari`@`localhost` IDENTIFIED BY '1234';
CREATE USER `ambari`@`%` IDENTIFIED BY '1234';
CREATE USER `hive`@`localhost` IDENTIFIED BY '1234';
CREATE USER `hive`@`%` IDENTIFIED BY '1234';
CREATE USER `oozie`@`localhost` IDENTIFIED BY '1234';
CREATE USER `oozie`@`%` IDENTIFIED BY '1234';

GRANT ALL PRIVILEGES ON *.* to `ambari`@`localhost` with grant option;
GRANT ALL PRIVILEGES ON *.* to `ambari`@`%` with grant option;
GRANT ALL PRIVILEGES ON *.* to `hive`@`localhost` with grant option;
GRANT ALL PRIVILEGES ON *.* to `hive`@`%` with grant option;
GRANT ALL PRIVILEGES ON *.* to `oozie`@`localhost` with grant option;
GRANT ALL PRIVILEGES ON *.* to `oozie`@`%` with grant option;

FLUSH PRIVILEGES;

CREATE DATABASE ambari;
CREATE DATABASE oozie;
CREATE DATABASE hive;
{% endhighlight %}

{% highlight bash %}
mysql -u ambari -p ambari < /var/lib/ambari-server/resources/Ambari-DDL-MySQL-CREATE.sql
{% endhighlight %}


### Configuring root account

여러모로 root로 계정으로 HDP를 설정하는것이 여러모로 이롭습니다.

**On Localhost**

SSH 를 먼저 설정해줍니다.<br>
포인트는 root계정으로 해야하며,  ssh root@localhost를 했을때 에러가 없어야 합니다.

{% highlight bash %}
$ sudo su
$ cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
{% endhighlight %}

**On AWS EC2**

만약 EC2라면 ubuntu계정의 authorized_keys값을 root의 authorized_keys값에 넣습니다.
{% highlight bash %}
$ sudo su
$ cat /home/ubuntu/.ssh/authorized_keys > /root/.ssh/authorized_keys
{% endhighlight %}

{% highlight bash %}
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
{% endhighlight %}

ssh -i private-key.pem root@localhost 을 했을때 접속이 잘되면 EC2에서도 암호를 물어보지 않고 접속이 되면 됩니다.



### Setting up Ambari

{% highlight bash %}
sudo ambari-server setup
{% endhighlight %}

root계정으로 ambari를 돌릴려면 n 을 누르고, 새로운 유저를 만들려면 y를 누릅니다.

{% highlight bash %}
Customize user account for ambari-server daemon [y/n] (n)? n
{% endhighlight %}

계속 알수 없는 에러가 날 경우(MySQL접속 등등) 다음의 파일을 확인해 봅니다. 

| Name | Command | 
|:-----|:--------|
| Ambari Server | sudo vi /etc/ambari-server/conf/ambari.properties | 
| Ambari Agent |  sudo vi /etc/ambari-agent/conf/ambari-agent.ini |




### Hostname Settings

/etc/hosts로 들어가서 public domain을 넣습니다.

{% highlight bash %}
52.192.233.209	ec2-52-192-233-209.ap-northeast-1.compute.amazonaws.com
{% endhighlight %}

이후 저장하고 나와서 다음의 명령어로 hostname을 변경해줍니다.<br>
ambari-agent conf를 들어가서 

{% highlight bash %}
sudo hostname ec2-52-192-233-209.ap-northeast-1.compute.amazonaws.com
{% endhighlight %}

**ambari-agent** 안의 hostname을 변경합니다.

{% highlight bash %}
sudo vi /etc/ambari-agent/conf/ambari-agent.ini

hostname=PRIVATE_DOMAIN_NAME
{% endhighlight %}

### Start Ambari

{% highlight bash %}
$ sudo ambari-server start
{% endhighlight %}

<span style="color:red">
실행시킨후 8080포트로 들어가면 Ambari Webpage를 볼 수 있습니다.<br>
기본 ID/Password는 admin/admin 입니다.
</span>


### Installing HDP on AWS

이곳부터는 너무 쉬워서 tutorial도 필요없지만, 몇가지 알아야할 사항만 적는다면.. 

* EC2 사용시 Install Options -> Target Hosts안에 **internal private DNS**를 적습니다.<br>
  ex)ip-172-31-8-106.ap-northeast-1.compute.internal<br>
  또한 SSH Private Key에다가는 .pem파일을 업로드 하고, SSH User는 ubuntu로 변경해줍니다.
 
<img src="{{ page.asset_path }}install-options.png" class="img-responsive img-rounded">



### Success


<img src="{{ page.asset_path }}ambari.png" class="img-responsive img-rounded">




### Removing Service

Ambari를 사용중에 있다가 도중에 Service를 삭제해야할때도 있습니다. 이경우 다음과 같이 합니다.

{% highlight bash %}
curl -u admin:admin -H "X-Requested-By: ambari" -X DELETE http://AMBARI-SERVER-ADDRESS:8080/api/v1/clusters/CLUSTER-NAME/services/SERVICE-NAME
{% endhighlight %}


ZEPPELIN 삭제하는 방법

{% highlight bash %}
curl -u admin:admin -H "X-Requested-By: ambari" -X DELETE http://localhost:8080/api/v1/clusters/test/services/ZEPPELIN
{% endhighlight %}


# HDP Configuration when using Public IP

### JAVA_HOME Configuration

JAVA_HOME을 설정해주어야 합니다. (ambari-metrics-collector에서 에러가 날 수 있습니다.)

{% highlight bash %}
sudo vi /etc/ambari-metrics-collector/conf/ams-env.sh
{% endhighlight %}

{% highlight bash %}
# Java
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
unset JAVA_TOOL_OPTIONS
{% endhighlight %}

### dfs.namenode.*-address

설치후 Ambari설치시 설정한 hostname으로 박혀있을텐데 이것을 0.0.0.0으로 바꿔줘야 합니다.<br>
자세한 사항은 Namenode: BindException을 참고..

### Hbase Master Bind Address

hbase-site.xml -> hbase.master.hostname (0.0.0.0) 추가

<img src="{{ page.asset_path }}hbase-master-address.png" class="img-responsive img-rounded">

/etc/hbase/conf/regionservers 에서 regionserver의 hostname을 변경합니다.  

{% highlight bash %}
sudo vi /etc/hbase/conf/regionservers 
{% endhighlight %}

Hbase master 그리고 regionserver 를 테스트를 위해 키고 끄는건 다음과 같이 합니다.

{% highlight bash %}
sudo -u hbase hbase master (start|stop|restart)
sudo -u hbase hbase regionserver
{% endhighlight %}

### Ambari Matrix Collector

일단 Ambari Matrix Collector는 2가지 모드가 있습니다. (embedded, distributed)
embedded의 경우는 Hbase를 집접 하나 띄워서 (기존의 Hbase말고.. Ambari Matrix Collector 가 사용할 Hbase) 
여기에 저장시키고, distributed는 HDFS에다가 저장을 합니다.

**Ambari -> Ambari Metrics -> Custom ams-hbase-site** 에서 다음을 추가시켜줍니다.

| Name | Value |
|:-----|:------|
| hbase.master.hostname       | 0.0.0.0 | 
| hbase.regionserver.hostname | 0.0.0.0 |


<img src="{{ page.asset_path }}custom-ams-hbase-site.png" class="img-responsive img-rounded">


만약 Zookeeper안에 /ams-hbase-unsecure 라는 znode가 없으면 만들어 줍니다.

{% highlight bash %}
zookeeper-client 
ls /
create /ams-hbase-unsecure mydata
{% endhighlight %}

 



# Errors

### mysql-connector-java Error (when installing Oozie)

이 부분은 Oozie설치할때 나타나는 문제인데.. 다른 버젼이 서로 충돌해서 생기는 문제입니다. 

{% highlight bash %}
resource_management.core.exceptions.Fail: Execution of '/usr/bin/apt-get -q -o Dpkg::Options::=--force-confdef --allow-unauthenticated --assume-yes install mysql-connector-java' returned 100.
{% endhighlight %}

이렇게 에러가 나오면 다음과 같이 해줍니다.<br> 
우분투에서 mysql-connector-java 를 설치할때 이미 설치된 라이브러리버젼과 달라 충돌해서 생기는 에러입니다.

{% highlight bash %}
sudo apt-get remove libmysql-java
{% endhighlight %}

그냥 지우기만 하면 안되고.. Oozie설치후 mysql-connector-java가 설치되었는지 확인해 봅니다. (즉 Oozie설치되면서 대체됨)

{% highlight bash %}
ls /usr/share/java | grep mysql-connector-java
{% endhighlight %}

### Agent Hostname Not Matching 

{% highlight bash %}
ERROR 2016-07-08 23:31:39,635 main.py:146 - Ambari agent machine hostname (localhost) does not match expected ambari server hostname (ip-172-31-27-227.ap-northeast-1.compute.internal). Aborting registration. Please check hostname, hostname -f and /etc/hosts file to confirm your hostname is setup correctly
{% endhighlight %}

이경우 Agent Configuration이 잘못된 경우입니다.

{% highlight bash %}
sudo vi /etc/ambari-agent/conf/ambari-agent.ini
{% endhighlight %}

아래의 hostname을 ambari 웹 컨트롤 페이지의 target hosts에서 적었던 hostname과 동일하게 맞춰줍니다.

{% highlight bash %}
[server]
hostname=ip-172-31-27-227.ap-northeast-1.compute.internal
url_port=8440
secured_url_port=8441
{% endhighlight %}

Agent를 Restart해줍니다.

{% highlight bash %}
ambari-agent restart
{% endhighlight %}

그 다음 hostname 명령어를 통해서 일치시켜줍니다.

{% highlight bash %}
sudo hostname ip-172-31-27-227.ap-northeast-1.compute.internal
{% endhighlight %}

### Internal Bug
 
{% highlight text %}
  File "/var/lib/ambari-agent/cache/stacks/HDP/2.0.6/hooks/before-START/scripts/params.py", line 158
    ambari_db_rca_password = config['hostLevelParams']['ambari_db_rca_password'][0]
TypeError: 'int' object has no attribute '__getitem__'
{% endhighlight %}

위와 같은 에러 발생시 /var/lib/ambari-agent/cache/stacks/HDP/2.0.6/hooks/before-START/scripts/params.py편집해서 158 line근처에 [0] 이 붙어있는 부분을 제거해줍니다.

{% highlight bash %}
sudo vi /var/lib/ambari-agent/cache/stacks/HDP/2.0.6/hooks/before-START/scripts/params.py
{% endhighlight %}

다음과 같이 만들어 줍니다.

{% highlight bash %}
ambari_db_rca_url = config['hostLevelParams']['ambari_db_rca_url']
ambari_db_rca_driver = config['hostLevelParams']['ambari_db_rca_driver']
ambari_db_rca_username = config['hostLevelParams']['ambari_db_rca_username']
ambari_db_rca_password = config['hostLevelParams']['ambari_db_rca_password']
{% endhighlight %}


### Ambari Agent Error

분명 apt-cache showpkg ambari-agent 하면 설치가 되어 있음에도 불구하고 sudo ambari-agent를 치면 없는 명령어라고 나오는 경우가 있을수 있습니다.
이 경우 manually ambari-agent를 설치해야 합니다.

{% highlight bash %}
WARNING: A HTTP GET method, public javax.ws.rs.core.Response org.apache.ambari.server.api.services.FeedService.getFeed(java.lang.String,javax.ws.rs.core.HttpHeaders,javax.ws.rs.core.UriInfo,java.lang.String), should not consume any entity.
{% endhighlight %}

{% highlight bash %}
sudo apt-get install ambari-agent
{% endhighlight %}


## Unable to locate package [package-name]

{% highlight bash %}
sudo vi /etc/apt/sources.list.d/ambari.list
{% endhighlight %}

열고 deb http://public-repo-1.hortonworks.com/ambari/ubuntu14/2.x/updates/2.2.2.0 Ambari main 이런 내용이 comment out 되진 않았는지 살펴봅니다.

### Oozie: Unauthorized connection for super-user

에러는 다음과 같습니다.

{% highlight bash %}
org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.security.authorize.AuthorizationException): Unauthorized connection for super-user: oozie from IP 172.31.31.188
{% endhighlight %}

{% highlight bash %}
sudo su hdfs
vi /etc/hadoop/conf/core-site.xml
{% endhighlight %}

core-size.xml을 열은 이후 다음과 같이 변경합니다.

{% highlight bash %}
<property>
  <name>hadoop.proxyuser.oozie.groups</name>
  <value>*</value>
</property>

<property>
  <name>hadoop.proxyuser.oozie.hosts</name>
  <value>*</value>
</property>
{% endhighlight %}

Oozie를 restart시키기 이전에 먼저,  HDFS먼저 restart시킵니다.

<img src="{{ page.asset_path }}oozie_proxy.png" class="img-responsive img-rounded">

### Namenode: BindException

{% highlight text %}
ERROR namenode.NameNode: Failed to start namenode.
java.net.BindException: Port in use: ec2-52-192-233-209.ap-northeast-1.compute.amazonaws.com:50070
{% endhighlight %}

현상은 port를 사용중에 있지 않는데, 해당 Port가 사용중이라고 나오는 케이스입니다.
이 경우 NameNode Bind 시키려는 IP를 0.0.0.0으로 바꿔줘야 합니다.

| Name | default value |
|:-----|:------|
| fs.defaultFS | 건들지 말것 (hostname으로 둘것) |
| dfs.namenode.http-address | 0.0.0.0:50070 |
| dfs.namenode.https-address | 0.0.0.0:50470 |
| dfs.namenode.rpc-address | 0.0.0.0:8020 |
| dfs.namenode.secondary.http-address | 0.0.0.0:50090 |

[HDFS default configurations][HDFS default configurations] 를 참조.

### Ambari Metrics Errors

{% highlight text %}
INFO org.apache.zookeeper.ClientCnxn: Opening socket connection to server localhost/127.0.0.1:61181. Will not attempt to authenticate using SASL (unknown error)
2016-08-03 04:30:41,131 WARN org.apache.zookeeper.ClientCnxn: Session 0x0 for server null, unexpected error, closing socket connection and attempting reconnect
java.net.ConnectException: Connection refused
	at sun.nio.ch.SocketChannelImpl.checkConnect(Native Method)
	at sun.nio.ch.SocketChannelImpl.finishConnect(SocketChannelImpl.java:717)
	at org.apache.zookeeper.ClientCnxnSocketNIO.doTransport(ClientCnxnSocketNIO.java:361)
	at org.apache.zookeeper.ClientCnxn$SendThread.run(ClientCnxn.java:1125)
{% endhighlight %}

위의 경우는 Hbase가 안띄워져 있어서 그렇습니다.


| Service | Name | default value |
|:--------|:-----|:------|
| Ambari Metrics | hbase.zookeeper.quorum | 0.0.0.0 |



[hortonworks hadoop with ambari]: http://docs.hortonworks.com/HDPDocuments/Ambari-2.2.2.0/bk_Installing_HDP_AMB/content/_download_the_ambari_repo_ubuntu14.html
[HDFS default configurations]: https://hadoop.apache.org/docs/r2.4.1/hadoop-project-dist/hadoop-hdfs/hdfs-default.xml
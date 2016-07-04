---
layout: post
title:  "Hortonworks Hadoop & Flamingo"
date:   2015-09-08 01:00:00
categories: "hadoop"
asset_path: /assets/posts/Hortonworks-Hadoop/

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


### Installing Ambari

[hortonworks hadoop with ambari][hortonworks hadoop with ambari] 링크를 참조

{% highlight bash %}
$ sudo wget -nv http://public-repo-1.hortonworks.com/ambari/ubuntu14/2.x/updates/2.2.2.0/ambari.list -O /etc/apt/sources.list.d/ambari.list

$ sudo apt-key adv --recv-keys --keyserver keyserver.ubuntu.com B9733A7A07513CAD
$ sudo apt-get update
$ sudo apt-get install ambari-server
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

### Setting up Ambari

{% highlight bash %}
sudo ambari-server setup
{% endhighlight %}

root계정으로 ambari를 돌릴려면 n 을 누르고, 새로운 유저를 만들려면 y를 누릅니다.

{% highlight bash %}
Customize user account for ambari-server daemon [y/n] (n)? n
{% endhighlight %}

계속 알수 없는 에러가 날 경우(MySQL접속 등등) 다음의 파일을 확인해 봅니다. 

{% highlight bash %}
sudo vi /etc/ambari-server/conf/ambari.properties
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

만약 SSH private Key가 필요하다면 다음과 같이 Private Key를 꺼낼수 있습니다.


### Installing HDP on Localhost

SSH 를 먼저 설정해줍니다. 

{% highlight bash %}
$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
# 또는 다음과 같이 해줍니다.
$ ssh-copy-id localhost
{% endhighlight %}

{% highlight bash %}
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
{% endhighlight %}

ssh localhost같이 접속을 할때 암호를 물어보지 않아도 접속이 되면 설정이 된 것입니다.<br>
**SSH Public Key (id_rsa.pub)은 target hosts의 root account 밑에 카피합니다.** 



[hortonworks hadoop with ambari]: http://docs.hortonworks.com/HDPDocuments/Ambari-2.2.2.0/bk_Installing_HDP_AMB/content/_download_the_ambari_repo_ubuntu14.html

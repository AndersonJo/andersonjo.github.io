---
layout: post
title:  "Zeppelin & Spark"
date:   2016-02-17 01:00:00
categories: "spark"
asset_path: /assets/posts/Zeppelin-Spark/
tags: ['hadoop', 'hortonworks']

---

<header>
<img src="{{ page.asset_path }}spark.jpg" class="img-responsive img-rounded" style="width:100%">
</header>

# Installation 

### Prerequisites 

* nodejs, npm 설치되어 있어야함
* JDK 설치 되어 있어야함
* Maven 설치 되어 있어야함

{% highlight bash %}
sudo apt-get install libfontconfig
{% endhighlight %}

### From Binary Package

먼저 [downloading pre-built binary package](http://zeppelin.apache.org/download.html)에서 다운로드 받습니다.<br>

{% highlight bash %}
sudo mv zeppelin* /usr/share/
sudo ln -s /usr/share/zeppelin-0.6.2-bin-all/ /usr/share/zeppelin
{% endhighlight %}
 
그 다음 interpreters를 설치해줍니다.
 
{% highlight bash %}
./bin/install-interpreter.sh --all
{% endhighlight %}

### Git Repository로 notebook사용하기 

conf디렉토리로 이동합니다. 

{% highlight bash %}
cd /usr/share/zeppelin/conf
cp zeppelin-site.xml.template zeppelin-site.xml
vi zeppelin-site.xml
{% endhighlight %}

다음의 내용을 uncomment 해줍니다.

{% highlight bash %}
<property>
  <name>zeppelin.notebook.storage</name>
  <value>org.apache.zeppelin.notebook.repo.GitNotebookRepo</value>
  <description>notebook persistence layer implementation</description>
</property>
{% endhighlight %}



### 실행하기 

**Daemon으로 실행및 종료**

{% highlight bash %}
bin/zeppelin-daemon.sh start
bin/zeppelin-daemon.sh stop
{% endhighlight %}

# Installing Zeppelin on HDP

### Installing in easy way

* [Hortonworks Zeppelin](http://hortonworks.com/apache/zeppelin/#section_3)

{% highlight bash %}
VERSION=`hdp-select status hadoop-client | sed 's/hadoop-client - \([0-9]\.[0-9]\).*/\1/'`
sudo git clone https://github.com/hortonworks-gallery/ambari-zeppelin-service.git /var/lib/ambari-server/resources/stacks/HDP/$VERSION/services/ZEPPELIN
{% endhighlight %}

설치 이후 ambari-server를 restart해줍니다.

{% highlight bash %}
sudo apt-get update
sudo ambari-server restart
{% endhighlight %}

Ambari에서 Add Service -> Zeppelin Notebook 을 통해서 설치가 가능합니다.


### Installing in hard way

{% highlight bash %}
sudo apt-get remove zeppelin*

cd ~/Downloads
git clone https://github.com/apache/zeppelin.git

sudo mkdir /var/lib/zeppelin
sudo mkdir -p /etc/zeppelin/conf.dist/
sudo cp zeppelin/conf/* /etc/zeppelin/conf.dist/
{% endhighlight %}




### Installing Zeppelin manually on HDP

* [Zeppelin Installation](https://github.com/apache/zeppelin)

{% highlight bash %}
git clone https://github.com/apache/zeppelin.git
cd zeppelin
mvn clean package -DskipTests [Options]
{% endhighlight %}

Examples 

{% highlight bash %}
mvn clean package -DsipTests -Pspark-1.6 -Phadoop-2.6 -Pyarn
{% endhighlight %}



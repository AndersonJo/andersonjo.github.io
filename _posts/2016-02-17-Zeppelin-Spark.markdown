---
layout: post
title:  "Zeppelin & Spark"
date:   2016-02-17 01:00:00
categories: "spark"
asset_path: /assets/posts/Zeppelin-Spark/
tags: ['hadoop', 'hortonworks']

---

<img src="{{ page.asset_path }}spark.jpg" class="img-responsive img-rounded">

# Installing Zeppelin 

### Prerequisites 

* nodejs, npm 설치되어 있어야함
* JDK 설치 되어 있어야함
* Maven 설치 되어 있어야함

{% highlight bash %}
sudo apt-get install libfontconfig
{% endhighlight %}


### Installing Zeppelin on HDP

* [Hortonworks Zeppelin][Hortonworks Zeppelin]

{% highlight bash %}
VERSION=`hdp-select status hadoop-client | sed 's/hadoop-client - \([0-9]\.[0-9]\).*/\1/'`
sudo git clone https://github.com/hortonworks-gallery/ambari-zeppelin-service.git /var/lib/ambari-server/resources/stacks/HDP/$VERSION/services/ZEPPELIN
{% endhighlight %}

설치 이후 ambari-server를 restart해줍니다.

{% highlight bash %}
sudo ambari-server restart
{% endhighlight %}

Ambari에서 Add Service -> Zeppelin Notebook 을 통해서 설치가 가능합니다.


### Installing Zeppelin manually on HDP

* [Zeppelin Installation][Zeppelin Installation]

{% highlight bash %}
git clone https://github.com/apache/zeppelin.git
cd zeppelin
mvn clean package -DskipTests [Options]
{% endhighlight %}

Examples 

{% highlight bash %}
mvn clean package -DsipTests -Pspark-1.6 -Phadoop-2.6 -Pyarn
{% endhighlight %}




[Zeppelin Installation]: https://github.com/apache/zeppelin
[Hortonworks Zeppelin]: http://hortonworks.com/apache/zeppelin/#section_3
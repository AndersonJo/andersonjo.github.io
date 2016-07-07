---
layout: post
title:  "Zeppelin & Spark"
date:   2016-02-17 01:00:00
categories: "spark"
asset_path: /assets/posts/Zeppelin-Spark/
tags: ['hadoop', 'hortonworks']

---

<img src="{{ page.asset_path }}spark.jpg" class="img-responsive img-rounded">

# Install Zeppelin 

### Before Build

* nodejs, npm 설치되어 있어야함
* JDK 설치 되어 있어야함
* Maven 설치 되어 있어야함

{% highlight bash %}
sudo apt-get install libfontconfig
{% endhighlight %}

### Build

{% highlight bash %}
git clone https://github.com/apache/zeppelin.git
cd zeppelin
mvn clean package -DskipTests [Options]
{% endhighlight %}
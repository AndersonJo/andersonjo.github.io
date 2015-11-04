---
layout: post
title:  "Elasticsearch 101"
date:   2015-11-04 01:00:00
categories: "analytics"
asset_path: /assets/posts/Elasticsearch-101/
---
<div>
    <img src="{{ page.asset_path }}logo-elastic.png" class="img-responsive img-rounded">
</div>

# Installing Elasticsearch

### Installing Elasticsearch

JDK가 먼저 깔려 있어야 하고, devian package로 다운받으면 됩니다.

{% highlight bash %}
dpkg -i elasticsearch-2.0.0.deb
{% endhighlight %}

### Running as a service

재부팅이 될때마다 바로 실행이 되버리면 잘못된 configuration으로 cluster이 join하게 될 수 있으므로 기본적으로 
집접 booted된 이후에 elasticsearch를 실행시키도록 만들어야 합니다.

{% highlight bash %}
sudo update-rc.d elasticsearch defaults 95 10
sudo /etc/init.d/elasticsearch start
{% endhighlight %}

Ubuntu14이상에서는 update-rc.d대신에 systemctl을 해줍니다.

{% highlight bash %}
sudo /bin/systemctl daemon-reload
sudo /bin/systemctl enable elasticsearch.service
sudo /bin/systemctl start elasticsearch.service
{% endhighlight %}


# Elasticsearch 101

### Run!

{% highlight bash %}
sudo service elasticsearch start
{% endhighlight %}


**아직 끝나지 않은 문서입니다. 시간나는대로 Tutorial을 쓰겠습니다.**


[http://localhost:9200][http://localhost:9200]


[http://localhost:9200]: http://localhost:9200
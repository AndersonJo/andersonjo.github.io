---
layout: post
title:  "Elasticsearch & Logstash"
date:   2015-11-04 01:00:00
categories: "elastic"
asset_path: /assets/posts2/Elasticsearch/
tags: ['logstash']
---
<header>
    <img src="{{ page.asset_path }}logo-elastic.png" class="img-responsive img-rounded">
</header>

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

{% highlight bash %}
sudo service elasticsearch start
{% endhighlight %}

### Running on Docker

{% highlight bash %}
docker pull elasticsearch
docker run --name elasticsearch -p 9200:9200 -p 9300:9300 -d elasticsearch
{% endhighlight %}

# Configuration

sudo vi /etc/elasticsearch/elasticsearch.yml 에서 host를 변경가능
{% highlight bash %}
network.host: 0.0.0.0
{% endhighlight %}



# Logstash

### Installation

{% highlight bash %}
wget -qO - https://packages.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
echo "deb http://packages.elastic.co/logstash/2.2/debian stable main" | sudo tee -a /etc/apt/sources.list
sudo apt-get update
sudo apt-get install logstash
{% endhighlight %}


### Configuration

/etc/logstash/conf.d/ 여기에다가 input_to_output.conf 같은 파일이름 형식으로 설정파일을 만듭니다.

# Elasticsearch with Python

{% highlight python %}
from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
{% endhighlight %}








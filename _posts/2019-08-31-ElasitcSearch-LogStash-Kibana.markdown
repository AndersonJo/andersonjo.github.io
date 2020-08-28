---
layout: post
title:  "ElasticSearch + LogStash + Kibana = ELK"
date:   2019-07-20 01:00:00
categories: "elasticsearch"
asset_path: /assets/images/
tags: ['folium', 'kepler', 'python', 'h3']
---


# Installation

### ELasticSearch

설치는 다음과 같이 합니다.

{% highlight bash %}
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-7.x.list
{% endhighlight %}



{% highlight bash %}
sudo apt-get update
sudo apt-get install elasticsearch
{% endhighlight %}


설치완료된 이후 `elasticsearch.yml` 을 오픈합니다. 

{% highlight bash %}
sudo vi /etc/elasticsearch/elasticsearch.yml
{% endhighlight %}

설정파일에서 다음을 uncomment 합니다.

{% highlight bash %}
network.host: localhost
{% endhighlight %}



ElasticSearch를 재시작합니다.

{% highlight bash %}
sudo systemctl start elasticsearch
{% endhighlight %}

컴퓨터 재시작마다 ES를 자동으로 시작하게 만들려면 다음을 실행합니다.

{% highlight bash %}
sudo systemctl enable elasticsearch
{% endhighlight %}


아래의 명령어로 테스트를 합니다.

{% highlight bash %}
$ curl -X GET "localhost:9200"

{
  "name" : "VhoP2aS",
  "cluster_name" : "elasticsearch",
  "cluster_uuid" : "OjUD0yddQKyajUlLbbN1jA",
  "version" : {
    "number" : "6.8.2",
    "build_flavor" : "default",
    "build_type" : "deb",
    "build_hash" : "b506955",
    "build_date" : "2019-07-24T15:24:41.545295Z",
    "build_snapshot" : false,
    "lucene_version" : "7.7.0",
    "minimum_wire_compatibility_version" : "5.6.0",
    "minimum_index_compatibility_version" : "5.0.0"
  },
  "tagline" : "You Know, for Search"
}
{% endhighlight %}


[http://localhost:9200](http://localhost:9200) 로 접속해서 잘 되는지 확인합니다.

### Kibana

{% highlight bash %}
sudo apt-get install kibana
{% endhighlight %}

시스템 시작시 kibana를 부르려면 다음과 같이 합니다.

{% highlight bash %}
sudo systemctl enable kibana
sudo systemctl start kibana
{% endhighlight %}

기본적으로 Kibana는 localhost를 listen합니다. <br>
외부 접근시 Nginx를 설정해서 Kibana와 proxy를 설정해주어야 합니다.

[http://localhost:5601](http://localhost:5601) 접속시 Kibana로 들어가지는지 확인합니다.


###  Logstash

{% highlight bash %}
sudo apt-get install logstash
{% endhighlight %}



### Terms

1. **Index**: RDBMS에서 database로 생각하면 됩니다.
2. **DataType**: document의 type
3. **ID**: Ducument ID
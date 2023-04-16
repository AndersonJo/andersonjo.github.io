---
layout: post
title:  "Installing Elasticsearch & Plugins"
date:   2015-11-04 01:00:00
categories: "elastic"
asset_path: /assets/images/Elasticsearch/
tags: ['logstash']
---
<header>
    <img src="{{ page.asset_path }}logo-elastic.png" class="img-responsive img-rounded img-fluid">
</header>

# Installing Elasticsearch

### Installing Elasticsearch

JDK가 먼저 깔려 있어야 하고, [오피셜 다운로드 페이지](https://www.elastic.co/downloads) 참고합니다. 

{% highlight bash %}
$ sudo apt-get install apt-transport-https
$ wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg
$ echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
$ sudo apt-get update 
$ sudo apt-get install elasticsearch
{% endhighlight %}

### Running as a service

{% highlight bash %}
$ sudo systemctl daemon-reload
$ sudo systemctl enable elasticsearch.service
$ sudo systemctl start elasticsearch.service
{% endhighlight %}


아래의 링크에서 제대로 설치됐는지 확인합니다.<br>
[http://localhost:9200/](http://localhost:9200/)


#### [Error]validation exception


다음과 같은 에러가 날 수 있습니다.

{% highlight bash %}
validation exception
bootstrap checks failed
max virtual memory areas vm.max_map_count [65530] likely too low, increase to at least [262144]
{% endhighlight %}

해결책은 다음을 실행시켜줍니다.

{% highlight bash %}
sudo sysctl -w vm.max_map_count=262144
{% endhighlight %}


### Running on Docker

{% highlight bash %}
docker pull elasticsearch
docker run --name elasticsearch -p 9200:9200 -p 9300:9300 -d elasticsearch
{% endhighlight %}

### Installing Kibana

[키바나 다운로드](https://www.elastic.co/downloads/kibana) 에서 DEB파일을 다운로드 받고 설치합니다.

{% highlight bash %}
sudo dpkg -i kibana-*.deb

sudo systemctl daemon-reload
sudo systemctl enable kibana.service
sudo systemctl start kibana.service
{% endhighlight %}

설정파일을 엽니다.

{% highlight bash %}
sudo vi /etc/kibana/kibana.yml
{% endhighlight %}

다음을 수정할수 있습니다. (Optional)

{% highlight bash %}
server.host: "0.0.0.0"
{% endhighlight %}

아래의 링크에서 확인을 합니다.<br>
[http://localhost:5601/](http://localhost:5601/)

### Installing Plugins

**Installing X-Pack**

* Elastic 5.0이후부터 **Marvel** 또한 X-pack의 일부입니다.
* Kibana 설치이후에 x-pack을 설치해야 합니다. (Elasticsearch에 x-pack설치 한번하고, Kibana에서도 동일하게 x-pack을 설치다시한번 더 해야합니다.)

{% highlight bash %}
sudo /usr/share/elasticsearch/bin/elasticsearch-plugin install x-pack
sudo /usr/share/kibana/bin/kibana-plugin install x-pack
{% endhighlight %}

> X-pack 설치 이후 Security가 가동되기 때문에 Kibana또는 Elasticsearch 접속시 Authentication이 필요합니다.<br>
> 아래의 Security 부분을 참조 합니다.


### Installing Filebeat 

Filebeat은 서버에서 떨구어진 log 파일들을 취합해서 logstash로 보냅니다. <br>
먼저 [다운로드](https://www.elastic.co/downloads/beats/filebeat)페이지에서 deb 파일을 다운받습니다.

{% highlight bash %}
sudo dpkg -i filebeat-*.deb
{% endhighlight %}



### Installing Logstash

먼저 Public Signing Key를 다운로드후 등록합니다. 

{% highlight bash %}
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
{% endhighlight %}

다음을 설치합니다.

{% highlight bash %}
sudo apt-get install apt-transport-https
{% endhighlight %}

Repository를 등록후 apt-get으로 설치합니다.

{% highlight bash %}
echo "deb https://artifacts.elastic.co/packages/5.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-5.x.list
sudo apt-get update
sudo apt-get install logstash
{% endhighlight %}




# Security Configuration

### Default Authentication

X-pack을 설치한 후 Authentication이 작동됩니다. 기본적으로 5.x 버젼이후에는 기본적인 User와 Role이 정의되어 있습니다. 

| Name | Password | Description | version |
|:-----|:---------|:------------|:--------|
| elastic | changeme | a built-in superuser <br>만약 User, Role을 Kibana에서 변경하려고 하면 elastic으로 접속후 변경해야 합니다. | 5.x |
| kibana | changeme | Elasticsearch에 접속하고 통신하기 위해서 필요한 계정 | 5.x |


### Kibana에서 Elasticsearch Authentication 설정

Kibana에서 Elasticsearch접속하기 위해서는 기본적으로 User 그리고 Password를 설정해주어야 합니다.<br>
"kibana" user의 password를 변경해준뒤 반드시 다음의 설정을 해주어야 Kibana가 제대로 동작합니다.<br>
만약 Kibana가 Elasticsearch로 Authentication Exception이 나면.. service kibana status 치면 문제없이 나오지만.. web상에서는 아무런 화면이 뜨지 않는 에러가 나게 됩니다.

{% highlight bash %}
sudo vi /etc/kibana/kibana.yml
{% endhighlight %}

{% highlight yml %}
elasticsearch.username: "kibana"
elasticsearch.password: "newpassword"
{% endhighlight %}


### Generate Symmetric Key

아래의 명령어는 symmetric key를 생성하며 클러스터안의 모든 노드들에 동일한 키가 복사되게 해놓습니다.

{% highlight bash %}
$ sudo ./syskeygen
Storing generated key in [/etc/elasticsearch/x-pack/system_key]...
Ensure the generated key can be read by the user that Elasticsearch runs as, permissions are set to owner read/write only
{% endhighlight %}

클러스터안의 모든 노드들에 다음과 같이 설정을 해줍니다.

{% highlight bash %}
sudo vi /etc/elasticsearch/elasticsearch.yml
{% endhighlight %}

{% highlight bash %}
xpack.security.audit.enabled: true
{% endhighlight %}


### Disable security

{% highlight bash %}
sudo vi   /etc/elasticsearch/elasticsearch.yml
{% endhighlight %}



# First Logstash Tutorial

먼저 tutorial로 사용될 [Apache Logs](https://download.elastic.co/demos/logstash/gettingstarted/logstash-tutorial.log.gz) 를 다운받은후 복사합니다.

{% highlight bash %}
sudo mv logstash-tutorial-dataset /var/log/
{% endhighlight %}

> 일반적으로 logstash와 filebeat는 각기 다른 machine위에서 돌아갑니다.

Filebeat을 설정해줍니다.

{% highlight bash %}
sudo vi /etc/filebeat/filebeat.yml
{% endhighlight %}

다음을 설정합니다.

{% highlight yml %}
filebeat.prospectors:
- input_type: log
  paths:
    - /var/log/logstash-tutorial-dataset
output.logstash:
  hosts: ["localhost:5043"]
{% endhighlight %}




{% highlight bash %}
xpack.security.enabled: false
{% endhighlight %}
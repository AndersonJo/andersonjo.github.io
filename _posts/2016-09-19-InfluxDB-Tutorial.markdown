---
layout: post
title:  "InfluxDB Tutorial"
date:   2016-09-19 01:00:00
categories: "database"
asset_path: /assets/posts2/DB/
tags: ['time-series', ]

---

<header>
    <img src="{{ page.asset_path }}grafana-iot.png" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small>
        SQLdmf asdfadfmasdfsdfi
    </small>
    </div>
</header>

# Installation on Ubuntu

### Using the Ubuntu Repository
 
[Installation Guide][Installation Guide] 를 참고. 

{% highlight bash %}
curl -sL https://repos.influxdata.com/influxdb.key | sudo apt-key add -
source /etc/lsb-release
echo "deb https://repos.influxdata.com/${DISTRIB_ID,,} ${DISTRIB_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/influxdb.list
{% endhighlight %}

설치및 서비스 실행 

{% highlight bash %}
sudo apt-get update && sudo apt-get install influxdb
sudo systemctl start influxdb
{% endhighlight %}

Synchronous run 시키려면 다음과 같이 합니다.

{% highlight bash %}
sudo influxd 
{% endhighlight %}


# Shell Tutorial

### Connecting from shell

{% highlight bash %}
$ influx
Visit https://enterprise.influxdata.com to register for updates, InfluxDB server management, and monitoring.
Connected to http://localhost:8086 version 1.0.0
InfluxDB shell version: 1.0.0
{% endhighlight %}

### Basic 101 

**새로운 데이터베이스 생성**
{% highlight bash %}
CREATE DATABASE test
{% endhighlight %}

**Inserting & Selecting**

{% highlight text %}
<measurement>[,<tag-key>=<tag-value>...] <field-key>=<field-value>[,<field2-key>=<field2-value>...] [unix-nano-timestamp]
{% endhighlight %}

| Name | In RDBMS | 
|:-----|:---------|
| measurement | Table |
| tag | Indexed Column |
| field | Not Indexed Column |


CPU라는 measurement(Table in RDBMS)에 host, region columns은 index시키고 value는 index없이 값을 넣겠다는 것. 
{% highlight bash %}
$ INSERT cpu,host=serverA,region=us_west value=0.64
$ INSERT cpu,host=serverA,region=japan value=0.22,exo=13
$ SELECT * FROM cpu
time			exo	host	region	value
1474257390186787897		serverA	us_west	0.64
1474258022478192898	13	serverA	japan	0.22
{% endhighlight %}

# Java Client Tutorial
 
### Creating & Deleting database

{% highlight java %}
influx.query(new Query("CREATE DATABASE " + DB_NAME, DB_NAME));
influx.deleteDatabase(TEST_DATABASE_NAME);
{% endhighlight %}

> influx.createDatabase(DB_NAME)도 가능하지만, 2.2 client의 경우 IF NOT EXISTS가 들어가는데 여기서 에러가 난다.<br> 
> 2.3에서는 IF NOT EXISTS가 제거된 상태 

### Enabling Batch

{% highlight java %}
influx.enableBatch(2000, 100, TimeUnit.MILLISECONDS);
{% endhighlight %}

Batch를 사용하면, write performance를 증가시킬수 있습니다.
actions 또는 flushDuration이 도달하게 되면 batch write 가 실행이 됩니다.

| Name | Example | Description |
|:-----|:--------|:------------|
| actions | 2000 | 수집할 actions의 갯수 (이상 넘어가면 batch write가 실행) |
| flushDuration | 100 | TimeUnit.MILLISECONDS 에 따라서 시간이 되면 batch write를 실행시키게 됩니다. |

### Simple Writing Recipe

위의 enableBatch() function을 실행시킨후, 아래의 코드를 실행시 batch write로 작동이 됩니다.

{% highlight java %}
Point point = Point.measurement(TEST_MEASUREMENT_NAME)
        .tag("host", "serverA")
        .tag("region", "asia")
        .addField("value", 0.484518)
        .addField("temperature", 34)
        .addField("status", true).build();

influx.write(TEST_DATABASE_NAME, "autogen", point);
{% endhighlight %}


[Installation Guide]: https://docs.influxdata.com/influxdb/v1.0/introduction/installation/
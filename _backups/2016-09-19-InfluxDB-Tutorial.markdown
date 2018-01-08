---
layout: post
title:  "InfluxDB Tutorial"
date:   2016-05-01 01:00:00
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

**Reading multiple series**

InfluxDB는 SQL과 많이 닮아있지만, 다른점들도 있습니다. 그중에 하나가 multiple series를 가져올수 있는 것 입니다.<br>
이게 중요한 이유는 Java Client에서 getSeries()함수를 부를때 아래처럼 여러개의 series를 불러오기 때문입니다.

{% highlight bash %}
$ select * from SENSOR, SENSOR2
name: SENSOR
------------
time	count_status	count_temperature	count_value
0	28		28			28

name: SENSOR2
-------------
time	count_status	count_temperature	count_value
0	7		7			7
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

### Reading multiple series and values

Java Client는 multiple series를 가져옵니다. (항상.. 하나의 measurement 명시할때도..)

{% highlight java %}
QueryResult queryResult = influx.query(new Query("SELECT * FROM SENSOR", DATABASE_NAME));
for (QueryResult.Result r : queryResult.getResults()) {
    r.getSeries().get(0).getValues().stream().forEach(System.out::println);
    System.out.println(r.getSeries().size());
}
{% endhighlight %}

결과 화면. 

{% highlight text %}
[2016-09-23T05:50:49.773Z, HostA, Korea, true, -13.0, 30.0]
[2016-09-23T05:50:49.773Z, HostB, Korea, true, 7.0, 18.0]
[2016-09-23T05:50:49.776Z, HostA, Korea, true, 37.0, 24.0]
...
{% endhighlight %}

### Convert Time to Date 

자바에서 time을 불러오면은 **"2016-09-19T05:42:44.545699072Z"** 같은 형식의 String으로 되어 있는것을 볼 수 있습니다.<br>
대략 다음과 같은 코드 String 을 Date로 변환할 수 있습니다.

{% highlight java %}
SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSSSSS'Z'");
series.getValues().stream().map(c -> {
    Date date = null;
    try {
        date = dateFormat.parse((String) c.get(0));
    } catch (ParseException e) {
        e.printStackTrace();
        return null;
    }
})
{% endhighlight %}

[Installation Guide]: https://docs.influxdata.com/influxdb/v1.0/introduction/installation/
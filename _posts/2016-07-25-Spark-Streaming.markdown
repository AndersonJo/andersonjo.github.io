---
layout: post
title:  "Spark Streaming"
date:   2016-07-23 01:00:00
categories: "spark"
asset_path: /assets/posts/Spark-Streaming/
tags: ['hadoop', 'kafka']

---

<div>
    <img src="{{ page.asset_path }}speed.jpg" class="img-responsive img-rounded">
</div>

# Introduction

Spark Streaming은 Core Spark의 extension으로서, 여러 sources로 부터 데이터를 받아서 처리할 수 있도록 도와줍니다.
Kafka, Flume, Twitter, ZeroMQ, Kinesis, 또는 TCP Sockets 등등으로 부터 받을 수 있습니다.

<img src="{{ page.asset_path }}streaming-arch.png" class="img-responsive img-rounded">

내부적으로는 다음과 같이 데이터를 받아서 처리하게 됩니다.<br>
Spark Streaming은 데이터 스트림을 받은후, 데이터를 Batches로 나누게 됩니다.<br>
이후 Spark Engine은 배치를 처리하게 됩니다.

<img src="{{ page.asset_path }}streaming-flow.png" class="img-responsive img-rounded">

### Installing SBT

### Maven & SBT Configuration

**Maven**

{% highlight xml %}
<dependency>
    <groupId>org.apache.spark</groupId>
    <artifactId>spark-streaming_2.10</artifactId>
    <version>1.2.0</version>
</dependency>
{% endhighlight %}

**SBT**

{% highlight bash %}
libraryDependencies += "org.apache.spark" % "spark-streaming_2.10" % "1.2.0"
{% endhighlight %}


| Source | Artifact |
|:-------|:---------|
| Kafka | spark-streaming-kafka_2.10 |
| Flume | spark-streaming-flume_2.10 |
| Kinesis | spark-streaming-kinesis-asl_2.10 [Amazon Software License] |
| Twitter | spark-streaming-twitter_2.10 |
| ZeroMQ | spark-streaming-zeromq_2.10 |
| MQTT | spark-streaming-mqtt_2.10 |


# Word Count Tutorial


### Spark Master & Slave

{% highlight bash %}
start-master.sh -h hostname
start-slave -h hostname
{% endhighlight %}

hostname을 0.0.0.0으로 쓰면 에러가 날수 있습니다. (String으로 쓸것)

| Name | Port | Description |
|:-----|:-----|:------------|
| Spark Master | 8081 | Spark Master 주소를 볼수 있습니다. |
| Spark Web Interface | 4040 | |
| History Server | 18080 | |



### build.sbt

{% highlight text %}
name := "WordCount"
version := "1.0"

scalaVersion := "2.10.5"
libraryDependencies += "org.apache.spark" % "spark-streaming_2.10" % "1.5.2"
libraryDependencies += "junit" % "junit" % "4.10"
{% endhighlight %}

### Scala Code

{% highlight scala %}
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}

object StreamingTutorial {

  def main(args: Array[String]): Unit = {
    // Spark Configuration
    val conf = new SparkConf()
    conf.setAppName("anderson-streaming-tutorial")
    conf.setMaster("spark://hostname:7077")
    conf.set("spark.ui.conf", "4045")

    // Streaming Context
    val ssc = new StreamingContext(conf, Seconds(1))

    // Create a DStream
    val lines = ssc.socketTextStream("hostname", 9099)
    val words = lines.flatMap(_.split(" "))
    val pairs = words.map(word => (word, 1))
    val wordCounts = pairs.reduceByKey(_ + _)

    wordCounts.print()
    ssc.start()
    ssc.awaitTermination()
  }
}
{% endhighlight %}

* 에러 발생시, 0.0.0.0으로 되어 있는 hostname을 String이 들어간 FQDN으로 바꿔줍니다.

### Compile & Submit

{% highlight bash %}
sbt package
spark-submit --class StreamingTutorial --master spark://sf-dev:7077 target/scala-2.10/wordcount_2.10-1.0.jar
{% endhighlight %}

다른 shell화면을 띄우고 확인을 합니다.

{% highlight bash %}
nc -l sf-dev 9099
{% endhighlight %}


### Result

{% highlight text %}
-------------------------------------------
Time: 1469389644000 ms
-------------------------------------------
(b,2)
(hello,1)
(apple,2)
(a,5)
(hi,2)
(c,2)
{% endhighlight %}

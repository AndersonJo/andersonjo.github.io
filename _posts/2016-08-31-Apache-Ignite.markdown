---
layout: post
title:  "Apche Ignite"
date:   2016-08-31 01:00:00
categories: "in-memory-platform"
asset_path: /assets/images/Ignite/
tags: ['MapReduce', 'Spark']

---

<header>
    <img src="{{ page.asset_path }}682604.jpg" class="img-responsive img-rounded img-fluid">
    <div style="text-align:right;"> 
    <small>초등학교때부터 컴퓨터 학원 다니고, 게임하면서 했었던 둠.. 초등학교 4학년때 3.5 floppy 7개들고, 용산에 2번 왕복해서 Doom2 받은 기억이 새록새록.. <br>  
    나중에 커서 이런 게임 만들어야지 막 꿈을 키우던 시절.. ㅎㅎ 근데.. Apache Ignite하고 뭔 상관이지??<br>
    </small>
    </div>
</header>

# Installation

### Building from Source

1. [Download Ignite][Download Ignite]에서 다운을 받습니다.
2. 압축해제후, IGNITE_HOME을 설정해줍니다. (끝에 / 가 안붙도록 합니다.)

{% highlight bash %}
export IGNITE_HOME=/home/anderson/apps/apache-ignite-1.7.0-src
{% endhighlight %}

{% highlight bash %} 
# Build In-Memory Data Fabric release (without LGPL dependencies)
$ mvn clean package -DskipTests
 
# Build In-Memory Data Fabric release (with LGPL dependencies)
$ mvn clean package -DskipTests -Prelease,lgpl
 
# Build In-Memory Hadoop Accelerator release
# (optionally specify version of hadoop to use)
$ mvn clean package -DskipTests -Dignite.edition=hadoop [-Dhadoop.version=X.X.X]

# Example
mvn clean package -DskipTests -Prelease,lgpl -Dignite.edition=hadoop -Dhadoop.version=2.7.3
{% endhighlight %}


# Get Started!

### Start from shell

{% highlight bash %}
bin/ignite.sh
{% endhighlight %}

configuration file을 잡으려면 다음과 같이 합니다.

{% highlight bash %}
bin/ignite.sh examples/config/example-cache.xml
{% endhighlight %}

### Maven Configuration

[Apache Ignite Maven Repository][Apache Ignite Maven Repository]에서 
Ignite-core 는 반드시 넣고, Ignite-spring (Optional)은 추가시킬수 있습니다. 

**Gradle Example**

{% highlight bash %}
// Apache Ignite
// https://mvnrepository.com/artifact/org.apache.ignite/ignite-core
compile group: 'org.apache.ignite', name: 'ignite-core', version: '1.5.0.final'
compile group: 'org.apache.ignite', name: 'ignite-spring', version: '1.5.0.final'

// https://mvnrepository.com/artifact/com.h2database/h2
compile group: 'com.h2database', name: 'h2', version: '1.0.60'
{% endhighlight %}

**SBT 1.5 Example**

{% highlight bash %}
// Apache Ignite
// https://mvnrepository.com/artifact/org.apache.ignite/ignite-core
libraryDependencies += "org.apache.ignite" % "ignite-core" % "1.5.0.final"
libraryDependencies += "org.apache.ignite" % "ignite-spring" % "1.5.0.final"

// https://mvnrepository.com/artifact/com.h2database/h2
libraryDependencies += "com.h2database" % "h2" % "1.0.60"
{% endhighlight %}

### IntelliJ Configuration

| VM Option |  **-DIGNITE_HOME=[IGNITE_HOME_PATH]** | 

### Cache Example

{% highlight bash %}
bin/ignite.sh examples/config/example-cache.xml
{% endhighlight %}

{% highlight java %}
try (Ignite ignite = Ignition.start("examples/config/example-cache.xml")) {
    IgniteCache<Integer, String> cache = ignite.getOrCreateCache("myCacheName");

    // Store keys in cache (values will end up on different cache nodes).
    for (int i = 0; i < 10; i++)
        cache.put(i, Integer.toString(i));

    for (int i = 0; i < 10; i++)
        System.out.println("Got [key=" + i + ", val=" + cache.get(i) + ']');
}
{% endhighlight %}

{% highlight text %}
Got [key=0, val=0]
Got [key=1, val=1]
Got [key=2, val=2]
Got [key=3, val=3]
Got [key=4, val=4]
Got [key=5, val=5]
Got [key=6, val=6]
Got [key=7, val=7]
Got [key=8, val=8]
Got [key=9, val=9]
{% endhighlight %}

### Distributed Compute Example(MapReduce)

각각의 node안에 있는 cache데이터를 꺼내와서 compute를 실행시킵니다. 

{% highlight bash %}
bin/ignite.sh
{% endhighlight %}

{% highlight java %}
Ignition.setClientMode(true);
try (Ignite ignite = Ignition.start("config/default-config.xml")) {

    // Compute실행이 오직 remote nodes에서만 돌아갑니다.
    ClusterGroup remoteClusterGroup = ignite.cluster().forRemotes();
    IgniteCompute clusterCompute = ignite.compute(remoteClusterGroup);

    // Initialize Cache
    IgniteCache<Integer, String> cache = ignite.getOrCreateCache("features");
    cache.removeAll();

    // Insert data into cache
    for (int i = 0; i < 10; i++)
        cache.put(i, "This is " + Integer.toString(i));

    // MapReduce in distributed way
    ArrayList<Integer> netResults;
    netResults = (ArrayList<Integer>) clusterCompute.broadcast(
            (IgniteClosure<Integer, Integer>) t -> {
                int result = StreamSupport.stream(cache.localEntries(CachePeekMode.PRIMARY).spliterator(), false)
                        .sorted((a, b) -> a.getKey().compareTo(b.getKey()))
                        .map((e) -> {
                            System.out.println(e.getKey() + " " + e.getValue());
                            return e.getKey() * t;
                        })
                        .limit(3)
                        .reduce((a, b) -> a + b).get();
                return result;
            }, 2);

    netResults.forEach(System.out::println);
}
{% endhighlight %}

**Node1**
{% highlight bash %}
0 * t = 0
4 * t = 8
6 * t = 12
{% endhighlight %}

**Node2**
{% highlight bash %}
1 * t = 2
2 * t = 4
3 * t = 6
{% endhighlight %}

**Client**
{% highlight bash %}
12
20
{% endhighlight %}

### LocalNode & Partitions

{% highlight java %}
ClusterNode localnode = ignite.cluster().localNode();
int[] partitions = ignite.affinity("cache-name").allPartitions(localnode);
{% endhighlight %}

### Node Count

Server로 띄워져있는 node들의 갯수를 알아보는 코드입니다.<br>
!isClient() 함수로 체크를 안할시, 클라이언트또한 Node로 포함이 됩니다.

{% highlight java %}
StreamSupport.stream(this.ignite.cluster().nodes().spliterator(), false)
        .filter((n) -> !n.isClient())
        .count();
{% endhighlight %}


# Errors

**BinaryObjectException: Cannot find schema for object with compact footer**

binaryConfiguration를 추가시키면 됩니다.

{% highlight xml %}
<bean id="grid.cfg" class="org.apache.ignite.configuration.IgniteConfiguration">
    <property name="binaryConfiguration">
        <bean class="org.apache.ignite.configuration.BinaryConfiguration">
            <property name="compactFooter" value="false"/>
        </bean>
    </property>
</bean>
{% endhighlight %}
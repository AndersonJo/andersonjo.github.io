---
layout: post
title:  "Apche Ignite"
date:   2016-08-31 01:00:00
categories: "hadoop"
asset_path: /assets/posts2/Ignite/
tags: ['MapReduce', 'Spark']

---

<div>
    <img src="{{ page.asset_path }}682604.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small>초등학교때부터 컴퓨터 학원 다니고, 게임하면서 했었던 둠.. 초등학교 4학년때 3.5 floppy 7개들고, 용산에 2번 왕복해서 Doom2 받은 기억이 새록새록.. <br>  
    나중에 커서 이런 게임 만들어야지 막 꿈을 키우던 시절.. ㅎㅎ 근데.. Apache Ignite하고 뭔 상관이지??<br>
    </small>
    </div>
</div>

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

### Compute Example

{% highlight bash %}
bin/ignite.sh
{% endhighlight %}

{% highlight java %}
try (Ignite ignite = Ignition.start("examples/config/example-ignite.xml")) {
    Collection<IgniteCallable<Integer>> calls = new ArrayList<>();

    // Iterate through all the words in the sentence and create Callable jobs.
    for (final String word : "Count characters using callable".split(" "))
        calls.add(word::length);

    // Execute collection of Callables on the grid.
    Collection<Integer> res = ignite.compute().call(calls);

    // Add up all the results.
    int sum = res.stream().mapToInt(Integer::intValue).sum();

    System.out.println("Total number of characters is '" + sum + "'.");
}
{% endhighlight %}

[Download Ignite]: http://ignite.apache.org/download.cgi
[Apache Ignite Maven Repository]: https://mvnrepository.com/artifact/org.apache.ignite
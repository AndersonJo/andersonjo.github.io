---
layout: post
title:  "Gradle, JUnit, Maven"
date:   2015-04-18 01:00:00
categories: "java"
static: /assets/posts/Gradle-Junit-Maven/
tags: []

---

<img src="{{ page.static }}gradle.png" class="img-responsive img-rounded">


# Tasks

### Running Multiple Tasks

{% highlight ruby %}
task compile << {
    println '컴파일링!'
}

task unittest(dependsOn: compile) << {
    println '유닛테스트!'
}
{% endhighlight %}

gradle compile unittest 이렇게해서 2개의 tasks를 실행시키지만, compile task가 중복되서 실행되지는 않습니다.<br>
dependsOn에 상관없이 gradle에서는 한번만 실행이 됩니다.

{% highlight bash %}
$ gradle compile unittest
:compile
컴파일링!
:unittest
유닛테스트!

BUILD SUCCESSFUL

Total time: 2.576 secs
{% endhighlight %}

gradle unittest -x compile 처럼 -x뒤에 task를 쓰면 **exclude**가 됩니다. <br>
(즉 compile task는 실행이 안되고 unittest만 실행됨)

### Obtaining build information

**gradle -q projects**<br>
서브 프로젝트 리스트를 보여줍니다. -q는  quiet <br>
description='something'을 넣을수 있습니다.

### The Gradle Daemon 

Build하는데 들어가는 bootstrapping이나 기타등등을 메모리에 계속 올려놓음으로서 빠르게 빌드 가능합니다.<br>
(개발시에서는 항상 켜놓고, CI에서는 항상 꺼놓는게 좋습니다.)

**~/.gradle/gradle.properties** 의 파일안에 다음을 집어넣고 저장합니다.

{% highlight bash %}
org.gradle.daemon=true
{% endhighlight %}


# Gradle Wrapper

### Executing a build with the Wrapper

**각각의 Gradle Wrapper는 특정 버젼과 연결이 되어 있습니다.** <br>
따라서 ./gradlew <task>를 실행시 일치하는 gradle version을 다운로드 하고 빌드에 사용합니다.<br>
즉 gradle 프로젝트를 다른 컴퓨터에서 실행시, 따로 gradle를 설치할 필요가 없습니다.

### Adding the Wrapper

wrapper를 추가하려면 gradle wrapper를 실행시키면 되며, 특정 버젼을 명시하기 위해서는 --gradle-version <version>을 사용합니다.

{% highlight bash %}
gradle wrapper --gradle-version 2.5
{% endhighlight %}

wrapper task를 추가함으로서, gradle wrapper 를 실행시킬때 추가적인 Customize를 할 수 있습니다.

{% highlight ruby %}
task wrapper(type: Wrapper) {
    println 'Wrapper를 만듭니다.'
    gradleVersion = '2.0'
}
{% endhighlight %}

gradle wrapper 를 실행후 다음의 파일들이 만들어지는데, 모두 빠짐없이 Version Control System (Git)에 추가되어야 합니다.

{% highlight bash %}
project/
  gradlew
  gradlew.bat
  gradle/wrapper/
    gradle-wrapper.jar
    gradle-wrapper.properties
{% endhighlight %}

Gradle버젼 추후 변경시, gralde wrapper를 다시 실행할 필요 없이 gradle-wrapper.properties 파일을 수정하면 됩니다.



# JVM Projects

<img src="{{ page.static }}coffee.jpg" class="img-responsive img-rounded">


### Maven Naming

Maven Repository로 디플로이할때 Gradle은 자동으로 POM 을 생성합니다.<br>
이때 groupId, artifactId, version and packaging elements 들이 POM을 만들때 사용됩니다.

| Name | Description | Example |
|:-----|:------------|:--------|
| GroupId | Maven의 모든 프로젝트 안에서 Uniquely Identify 하는 이름. <br>주로 도메인 명을 사용합니다. | org.apache.maven <br>org.apache.commons |
| artifactId | 버젼이 안써진 Jar 파일 이름. <br>lowercase 알페벳으로 쓰는게 컨벤션 |  maven, commons-math | 
| version | 버젼.. | 2.0, 2.0.1, 1.3.1 |

### Basic JAVA Project

apply plugin 'java'를 해주면 Java Project의 기본적으로 사용하는 많은 tasks들이 자동으로 붙습니다.

build.gradle 예제..

{% highlight ruby %}
group 'andersonjo'
version '1.0-SNAPSHOT'
description = 'Anderson Java Note Book'

apply plugin: 'java'

sourceCompatibility = 1.5

repositories {
    mavenCentral()
}

dependencies {
    compile 'commons-codec:commons-codec:1.10'
    compile 'commons-io:commons-io:2.4'
    compile 'org.apache.commons:commons-lang3:3.0'
    compile 'com.google.guava:guava:19.0'
    testCompile group: 'junit', name: 'junit', version: '4.11'
}
{% endhighlight %}


# Maven

<img src="{{ page.static }}maven.png" class="img-responsive img-rounded">

Maven은 Project Object Model (POM)에 기초한 프로젝트 관리툴입니다. 

### Installation

apt-get maven으로 설치가 가능합니다. (현재 버젼 3.3.3)<br>
이때 maven2를 설치하면 안됩니다. (낮은 버젼이 설치됨)

{% highlight bash %}
sudo apt-get install maven
{% endhighlight %}

### Project Ojbect Model (POM)

groupId, artifactId, version 이렇게 3개가 repository안에서 다른 프로젝트들과 구별되게 만듭니다.

**pom.xml example**

{% highlight xml %}
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>io.github.andersonjo</groupId>
    <artifactId>maven-tutorial</artifactId>
    <version>0.0.1</version>

</project>
{% endhighlight %}
 
### Build Life Cycle

Build Lifecycle 은 다음과 같은 절차(sequence)를 밟게 됩니다.

<img src="{{ page.static }}maven-phase.png" class="img-responsive img-rounded">




# JUnit

<img src="{{ page.static }}tested.png" class="img-responsive img-rounded">

### Configuration

build.gradle 파일에 다음을 추가해 줍니다.

{% highlight ruby %}
apply plugin: 'java'

dependencies {
  testCompile 'junit:junit:4.12'
}

test {
    testLogging.showStandardStreams = true
}
{% endhighlight %}

 

{% highlight java %}
# src/test/java/hello/BasicTest.java
package hello;

import org.junit.Assert;
import org.junit.Test;

public class BasicTest {
    @Test
    public void testSample() {
        Assert.assertEquals("Gradle is awesome", "Gradle is awesome");
    }
}
{% endhighlight %}

전체 모든 테스트 케이스들을 다 돌리는 것은 다음과 같이 합니다.

{% highlight bash %}
./gradle test
{% endhighlight %}



[maven download]: https://maven.apache.org/download.cgi
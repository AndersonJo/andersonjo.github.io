---
layout: post
title:  "Spring Boot & Tomcat 101"
date:   2016-09-02 01:00:00
categories: "spring"
asset_path: /assets/posts2/Spring/
tags: ['java', 'Security', 'Tomcat']

---


<header>
    <img src="{{ page.asset_path }}dew.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small>
    Spring.. Java필요하거나 SI아니면.. 쓰레기.. 쓰지마세요. Python, Node.js에 정말 좋은 프레임워크많아요.<br>
     You Fool! Run away from this!
    </small>
    </div>
</header>

# Installation

### Tomecat 8 on Ubuntu 15.10

{% highlight bash %}
sudo apt-get install tomcat8 tomcat8-docs tomcat8-admin tomcat8-examples
{% endhighlight %}

다음을 추가 합니다. 
{% highlight bash %}
$ sudo vi /etc/tomcat8/tomcat-users.xml
  <role rolename="admin"/>
  <user username="tomcat" password="1234" roles="admin"/>
{% endhighlight %}

**Change configuration directory for Intellij**

{% highlight bash %}
mkdir -p ~/tomcat/conf/
sudo cp /etc/tomcat8/server.xml ~/tomcat/conf/
sudo cp /etc/tomcat8/web.xml ~/tomcat/conf/
sudo chown anderson:anderson ~/tomcat/conf/*
{% endhighlight %}

**Chaging default ports**

{% highlight bash %}
sudo vi /etc/tomcat8/server.xml
{% endhighlight %}

### Jetty on IntelliJ IDEA

{% highlight bash %}
sudo apt-get install jetty8
{% endhighlight %}


### Tomcat on IntelliJ IDEA

settings ->Build, Execution, Deployment -> Application Servers -> Tomcat 추가

| Name | Value | ETC | 
|:-----|:------|:----|
| Tomcat or TomEE Home | /usr/share/tomcat8 | | 
| Tomcat or TomEE directory | /home/anderson/tomcat | 해당 디렉토리 안의 conf 디렉토리를 찾음 | 

<img src="{{ page.asset_path }}settings-tomcat.png" class="img-responsive img-rounded">

Run -> Edit Configuration -> Add New Configuration -> Tomcat -> Application Server를 settings에서 미리 설정한 Tomcat Server를 선택

<img src="{{ page.asset_path }}tomcat-run-configuration.png" class="img-responsive img-rounded">

추가적으로 Deployment를 설정하고 artifact를 선택하면 됩니다.<br>
이때 exploded war를 사용해야지 변경된 부분만 업데이트하거나 Auto deploy가 가능해집니다.

**pom.xml**

{% highlight xml %}
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>io.anderson</groupId>
    <artifactId>spring-tutorial</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>war</packaging>

    <name>spring-tutorial</name>
    <description>Anderson's Spring Tutorial</description>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>1.4.1.RELEASE</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>

    <properties>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <project.reporting.outputEncoding>UTF-8</project.reporting.outputEncoding>
        <java.version>1.8</java.version>
        <tomcat.version>8.0.28</tomcat.version>
    </properties>

    <dependencies>
        <!-- Spring Boot -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>

        <!-- Tomcat -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-tomcat</artifactId>
            <scope>provided</scope>
        </dependency>
        <dependency>
            <groupId>org.apache.tomcat.embed</groupId>
            <artifactId>tomcat-embed-core</artifactId>
            <version>${tomcat.version}</version>
        </dependency>
        <dependency>
            <groupId>org.apache.tomcat.embed</groupId>
            <artifactId>tomcat-embed-logging-juli</artifactId>
            <version>${tomcat.version}</version>
        </dependency>
        <dependency>
            <groupId>org.apache.tomcat.embed</groupId>
            <artifactId>tomcat-embed-jasper</artifactId>
            <version>${tomcat.version}</version>
        </dependency>
        <dependency>
            <groupId>org.apache.tomcat</groupId>
            <artifactId>tomcat-jasper</artifactId>
            <version>${tomcat.version}</version>
        </dependency>
        <dependency>
            <groupId>org.apache.tomcat</groupId>
            <artifactId>tomcat-jasper-el</artifactId>
            <version>${tomcat.version}</version>
        </dependency>
        <dependency>
            <groupId>org.apache.tomcat</groupId>
            <artifactId>tomcat-jsp-api</artifactId>
            <version>${tomcat.version}</version>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>

    <repositories>
        <repository>
            <id>spring-releases</id>
            <url>https://repo.spring.io/libs-release</url>
        </repository>
    </repositories>
    <pluginRepositories>
        <pluginRepository>
            <id>spring-releases</id>
            <url>https://repo.spring.io/libs-release</url>
        </pluginRepository>
    </pluginRepositories>
</project>
{% endhighlight %}

**io.anderson.Application.java**
{% highlight java %}
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
{% endhighlight %}

# Useful Things

### Continuous Auto-Restart

**build.gradle** <small>dev configuration 그리고 spring-boot-devtools를 설치해야 합니다.</small>
{% highlight bash %}
configurations {
    dev
}

dependencies {
    dev("org.springframework.boot:spring-boot-devtools")
}

bootRun {
    // Use Spring Boot DevTool only when we run Gradle bootRun task
    classpath = sourceSets.main.runtimeClasspath + configurations.dev
}
{% endhighlight %}

**Build** <small>Intellij에서는 Alt + F12. 다음 2개의 명령어를 서로 다른 terminal에서 실행시켜야 합니다.</small>

{% highlight bash %}
./gradlew build --continuous --stacktrace
./gradlew bootRun
{% endhighlight %}

### Jetty instead of Tomcat

spring-boot-starter-web은 Tomcat을 embedded container를 기본으로 사용합니다. 이것 대신에 Jetty를 넣을수 있습니다.

{% highlight bash %}
compile("org.springframework.boot:spring-boot-starter-web") {
    exclude module: "spring-boot-starter-tomcat"
}
compile("org.springframework.boot:spring-boot-starter-jetty")
{% endhighlight %}




# Recipes 

### Restful Controller Example

**Data Class**
{% highlight bash %}
public class Greeting {
    public final int id;
    public String content;

    public Greeting(int id, String content) {
        this.id = id;
        this.content = content;
    }
}
{% endhighlight %}


**Controller**
{% highlight java %}
@RestController
public class GreetingController {

    private static final String template = "Hello, %s!!";
    private final AtomicInteger counter = new AtomicInteger();

    @RequestMapping(value = "/greeting", method = RequestMethod.GET)
    public Greeting greeting(@RequestParam(value = "name", defaultValue = "World", required = true) String name) {
        return new Greeting(counter.incrementAndGet(), String.format(template, name));
    }
}
{% endhighlight %}

**http://localhost:8080/greeting?name=Anderon**
{% highlight json %}
{"id":1,"content":"Hello, Anderon!!"}
{% endhighlight %}

# Spring Security 

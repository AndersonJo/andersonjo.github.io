---
layout: post
title:  "Installing Hadoop 2.7 on Ubuntu 15.04"
date:   2015-09-08 01:00:00
categories: "hadoop"
asset_path: /assets/posts/Installing-Hadoop-On-Ubuntu/
---
<div>
    <img src="{{ page.asset_path }}server.jpg" class="img-responsive img-rounded">
</div>

# Preprequisites

### Installing Oracle Java 

OpenJDK는 퍼포먼스가 Oracle Java에 비해서 늦습니다. 
Oracle Java 8을 설치하도록 하겠습니다. <br>
[Hadoop Supported Java version][supported-java]

{% highlight bash %}
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer
{% endhighlight %}


### Installing requisite softwares

{% highlight bash %}
sudo apt-get install ssh rsync
{% endhighlight %}

### Adding Hadoop User (Optional)

보안, 백업, 관리 등등의 이유로 Hadoop 유저를 새로 생성하는 것이 좋습니다. 
물론 requirement 는 아닙니다.

{% highlight bash %}
sudo addgroup hadoop
sudo adduser --ingroup hadoop hduser
sudo adduser hduser sudo
{% endhighlight %}

### Configuring ssh and sshd

하둡은 ssh를 통해서 각각의 노드들을 관리합니다.
2번째 문장이 key pair를 만들게 되는데 패스워드는 없는 것으로 설정을 합니다. 
보안이 걱정된다면 암호를 넣는게 맞고, 하둡이 노드에 연결될때마다 패스워드 넣어야 하는 귀차니즘이 걱정되면 넣지 않으면 됩니다.

{% highlight bash %}
su hduser
ssh-keygen -t rsa -P ""
cat $HOME/.ssh/id_rsa.pub >> $HOME/.ssh/authorized_keys
{% endhighlight %}

다음의 명령어로 제대로 작동을 하는지 확인을 합니다.

{% highlight bash %}
ssh localhost
{% endhighlight %}

만약에 Connection Refused가 나면 ssh server 가 설치가 안되어 있어서 그럴수도 있습니다.<br>
또는 **/etc/ssh/sshd_config** 설정을 확인해보시면 됩니다.
{% highlight bash %}
sudo apt-get install openssh-server
{% endhighlight %}


# Hadoop Installation

### Installation from Source

다음을 설치해줍니다.

{% highlight bash %}
sudo apt-get install maven libssl-dev build-essential pkgconf cmake
{% endhighlight %}

protobuf 는 반드시 2.5 여야 합니다. <br>
protoc --version  버젼이 2.5 초과라면 다음과 같이 2.5를 설치합니다.

{% highlight bash %}
wget http://protobuf.googlecode.com/files/protobuf-2.5.0.tar.gz 
tar xzvf protobuf-2.5.0.tar.gz
cd  protobuf-2.5.0
./configure
make
make check
sudo make install
sudo ldconfig
protoc --version 
{% endhighlight %}

[Hadoop Download][hadoop-download] 

다운로드 페이지에서 hadoop-2.7.2-src.tar.gz 파일을 다운로드 합니다. (소스 코드 파일)
압축을 해제시키고 압축을 해제한 폴더로 들어갑니다.

* package 는  build 명령어
* -Pdist는 native code, documentation 없이 build하라는 뜻
* -Pdist,native 는 native code와 함께 build하라는 뜻
* -Pdist,native,docs 는 native code 그리고 documentation과 함께 빌드하라는 뜻
* -DskipTests 테스트를 skip
* -Dtar 는 tar파일을 만듭니다.

**hduser로 로그인해서 해야합니다. 반드시!**

{% highlight bash %}
su hduser
tar xvf hadoop-2.7.2-src.tar.gz
cd hadoop-2.7.2-src
mvn package -Pdist,native -DskipTests -Dtar
sudo cp -R hadoop-dist/target/hadoop-2.7.2 /usr/local/
sudo chown -R hduser:hadoop /usr/local/hadoop-2.7.2
{% endhighlight %}

빌드가 끝난후 hadoop-dist/target/hadoop-2.7.2 디렉토리를 /usr/local 에다가 복사합니다.<br>
또는 이미 설치가 되있는 상태라면, hadoop-dist/target/hadoop-2.7.2/lib/native 안의 내용물만 복사하면 됩니다.

확인 하는 방법..
{% highlight bash %}
hadoop checknative -a
{% endhighlight %}

다른 방법...

{% highlight bash %}
hduser:hadoop>file ./lib/native/libhadoop.so.1.0.0 
./lib/native/libhadoop.so.1.0.0: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, BuildID[sha1]=cb34de90c6ae1192cf393441f9a5e0f6f0465e7c, not stripped
{% endhighlight %}



### .bashrc

다음의 명령어들을 .bashrc에 넣어주시면 됩니다. (설정값들은 변경해주셔야 합니다.)

> 참고로 HADOOP_HOME 은 deprecated 되었습니다.<br>
> 대신에 HADOOP_PREFIX를 사용합니다.

{% highlight bash %}
# Java
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
unset JAVA_TOOL_OPTIONS

# Hadoop
export HADOOP_PREFIX=/usr/local/hadoop-2.7.2
export HADOOP_MAPRED_HOME=$HADOOP_PREFIX
export HADOOP_COMMON_HOME=$HADOOP_PREFIX
export HADOOP_HDFS_HOME=$HADOOP_PREFIX
export YARN_HOME=$HADOOP_PREFIX
export HADOOP_CONF_DIR=$HADOOP_PREFIX/conf
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_PREFIX/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_PREFIX/lib/native"
export HADOOP_CLASSPATH=$HADOOP_PREFIX/conf
export CLASSPATH=$CLASSPATH:$HADOOP_PREFIX/lib/*:.
export PATH=$PATH:$HADOOP_PREFIX/bin
export PATH=$PATH:$HADOOP_PREFIX/sbin
{% endhighlight %}



# Hadoop Configuration


### Standalone Mode 
기본적으로 하둡은 non-distributed mode 즉 single Java process로 돌아가도록 설정이 이미 되어 있습니다.<br>
Standalone Mode는 개발, 디버깅, 테스팅을 위해서 유용합니다.<br>
먼저 기본적으로 제공되는 conf파일들을 모두 설치된 하둡으로 카피해줍니다.<br>
(이때 바로 conf디렉토리가 없으면 만들어줍니다.)

{% highlight bash %}
# hduser인 상태에서..
cp hadoop-dist/target/hadoop-2.7.2/etc/hadoop/*.xml /usr/local/hadoop-2.7.2/conf
{% endhighlight %}

그 다음 아래의 파일들과 동일하게 설정을 해줍니다.

* [core-site.xml][conf-core-site.xml]
* [hdfs-site.xml][conf-hdfs-site.xml]
* [conf-hadoop-env.sh][conf-hadoop-env.sh]
* [mapred-site.xml][mapred-site.xml]


최소한 $HADOOP_PREFIX/conf/core-site.xml 그리고 conf/hadoop-env.sh 가 존재해야 합니다.

### conf/hadoop.env.sh

JAVA_HOME에 대한 경로를 변경시켜주세요. 

{% highlight bash %}
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
{% endhighlight %}


### conf/core-site.xml

여기에서 포트 설정, 데이터 파일 저장 위치 등등의 주요 설정들을 할수 있습니다.<br>
설정은 key-value pair로 이루어집니다.  또한 final 의 의미는 user application에 의해서 설정값이  overriden 되지 않도록 설정하는 것입니다.

{% highlight xml %}
<configuration>
    <property>
        <name>fs.default.name</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
{% endhighlight %}


### conf/hdfs.site.xml

자세한 내용은 [hdfs-default.xml][hdfs-wiki] 문서를 봐주세요 :)

{% highlight xml %}
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>/home/hduser/dfs/namenode</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/home/hduser/dfs/datanode</value>
    </property>
    <property>
        <name>dfs.permissions</name>
        <value>false</value>
    </property>
</configuration>
{% endhighlight %}

| Name | Value |
|:-----|:------|
| fs.default.name | 클러스터의 URI주소입니다. |
| dfs.data.dir | DataNode가 어디에 Data를 저장시킬지에 대한 경로입니다. |
| dfs.name.dir | NameNode metadata가 저장되는 위치 입니다. |
| dfs.replication | replication의 숫자이고 기본값으로 3으로 지정되어 있습니다. <br>이보다 더 작은 숫자는 reliability에 문제가 될 수 있습니다. |
| dfs.permissions | 기본값이 true이고, false이면 hduser 뿐만 아니라 모든 유저가 hdfs사용 가능합니다. |


# Standalone Mode

먼저 filesystem 을 format시켜줍니다. (최소한 최초 한번은 format이 필요합니다. 안하면 에러납니다.)

{% highlight bash %}
rm -Rf /tmp/hadoop-hduser
hdfs namenode -format
{% endhighlight %}

start-dfs.sh를 실행시켜서 NameNode, Secondary NameNode를 실행시킵니다.<br>
start-yarn.sh 는 ResourceManager를 실행시킵니다.

{% highlight bash %}
$ start-dfs.sh && start-yarn.sh
$ jps
19011 SecondaryNameNode
20691 Jps
18773 NameNode
19230 ResourceManager
{% endhighlight %}

DataNode, Tasktracker는 daemone으로 다음과 같이 실행시키거나 멈출수 있습니다.

{% highlight bash %}
# 시작하기
hadoop-daemon.sh start datanode
hadoop-daemon.sh start tasktracker

# 멈추기
hadoop-daemon.sh stop datanode
hadoop-daemon.sh stop tasktracker
{% endhighlight %}


여기서 문제는 DataNode가 보이지가 않는데 이 경우 집접 DataNode를 실행시켜서 문제를 확인해볼수 있습니다.

{% highlight bash %}
$ hdfs datanode
{% endhighlight %}


| Server | URL |
|:-------|:----|
| NameNode | <a href="http://localhost:50070" target="_blank">http://localhost:50070</a> |
| Secondary NameNode | <a href="http://localhost:50090/" target="_blank">http://localhost:50090/</a> |





# Pseudo-Distributed Mode 

Mapreduce job은 YARN에서 pseudo-distributed mode로 실행시킬수도 있습니다.

### yarn-site.xml

{% highlight xml %}
<?xml version="1.0"?>
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
</configuration>
{% endhighlight %}


### conf/mapred-site.xml

{% highlight xml %}
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
   <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
   </property>
</configuration>
{% endhighlight %}

### Start 

{% highlight bash %}
start-yarn.sh
{% endhighlight %}


          
[supported-java]: http://wiki.apache.org/hadoop/HadoopJavaVersions
[hadoop-download]: http://www.apache.org/dyn/closer.cgi
[hdfs-wiki]: http://hadoop.apache.org/docs/r2.4.1/hadoop-project-dist/hadoop-hdfs/hdfs-default.xml

[conf-core-site.xml]: {{ page.asset_path }}core-site.xml
[conf-hdfs-site.xml]: {{ page.asset_path }}hdfs-site.xml
[conf-hadoop-env.sh]: {{ page.asset_path }}hadoop-env.sh
[conf-yarn-site.sh]: {{ page.asset_path }}yarn-site.xml
[mapred-site.xml]: {{ page.asset_path }}mapred-site.xml
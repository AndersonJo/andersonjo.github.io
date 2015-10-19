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

## Preprequisites

#### Installing Oracle Java 

OpenJDK는 퍼포먼스가 Oracle Java에 비해서 늦습니다. 
Oracle Java 7을 설치하도록 하겠습니다. <br>
[Hadoop Supported Java version][supported-java]

{% highlight bash %}
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java7-installer

{% endhighlight %}

#### Adding Hadoop User (Optional)

보안, 백업, 관리 등등의 이유로 Hadoop 유저를 새로 생성하는 것이 좋습니다. 
물론 requirement 는 아닙니다.

{% highlight bash %}
sudo addgroup hadoop
sudo adduser --ingroup hadoop hduser
sudo adduser hduser sudo
{% endhighlight %}



#### Configuring ssh and sshd

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


## Hadoop Installation

#### Installation

> 경고: 하둡은 32비트로 distribute되고 있습니다. 그냥 사용해도 무관하지만 약간의 warning을 볼수 있습니다.<br>
>      완벽하게 설치하고 싶다면 아래의 "util.NativeCodeLoader Error" 를 참고해주세요 <br>
> 소스 파일로 설치하는 법이 나와 있습니다.
 
[Hadoop Download Page][hadoop-download] 에 들어가서 하둡을 다운받으면 됩니다.<br>
다운을 받고 압축해제후 원하는 폴더로 이동시켜줍니다. (저는 /usr/local/hadoop 에 설치했습니다.)<br>
그 이후에 hduser가 사용할수 있도록 권한을 변경시켜줍니다.

{% highlight bash %}
sudo chown -R hduser:hadoop hadoop-2.7.1/
{% endhighlight %}

#### .bashrc

다음의 명령어들을 .bashrc에 넣어주시면 됩니다. (설정값들은 변경해주셔야 합니다.)

{% highlight bash %}

export JAVA_HOME=/usr/lib/jvm/java-7-oracle
export HADOOP_HOME=/usr/local/hadoop-2.7.1
export HADOOP_CONF_DIR=$HADOOP_HOME/conf
export HADOOP_CLASSPATH=/usr/local/hadoop-2.7.1/conf
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib/native"
export PATH=$PATH:$HADOOP_HOME/bin
unset JAVA_TOOL_OPTIONS
{% endhighlight %}



## Hadoop Configuration

실제 풀어보면 conf디렉토리가 missing인 상태입니다. 하둡 다운로드 페이지에서 hadoop-2.7.1.tar.gz 파일을 다운로드 받으셨을텐데..
이것 이외에도 src가 더 붙은 hadoop-2.7.1-src.tar.gz  이런 파일이 있는데 여기안에 설정파일들이 들어있습니다.

압축을 풀고 **hadoop-2.7.1-src/hadoop-common-project/hadoop-common/src/main/conf** 에 가보면 필요한 설정파일들이 존재합니다.

* [core-site.xml][conf-core-site.xml]
* [hdfs-site.xml][conf-hdfs-site.xml]
* [conf-hadoop-env.sh][conf-hadoop-env.sh]

최소한 $HADOOP_HOME/conf/core-site.xml 그리고 conf/hadoop-env.sh 가 존재해야 합니다.

#### conf/hadoop.env.sh

JAVA_HOME에 대한 경로를 변경시켜주세요. 

{% highlight bash %}
export JAVA_HOME=/usr/lib/jvm/java-7-oracle
{% endhighlight %}


#### conf/core-site.xml

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


#### conf/hdfs.site.xml

자세한 내용은 [hdfs-default.xml][hdfs-wiki] 문서를 봐주세요 :)

{% highlight xml %}
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>3</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>/home/hduser/hdfs/name</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/home/hduser/hdfs/data</value>
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


#### Starting HDFS

먼저 filesystem 을 format시켜줍니다. (최소한 최초 한번은 format이 필요합니다. 안하면 에러납니다.)

{% highlight bash %}
hdfs namenode -format
{% endhighlight %}

start-dfs.sh 를 실행시킵니다. 

{% highlight bash %}
sbin/start-dfs.sh
{% endhighlight %}

실행하고 난뒤 localhost:50070 으로 들어가면 Hadoop Web Interface에 접근할수 있습니다. 
실행하고 난뒤 로그는 /logs 디렉토리안에서 확인가능합니다.

**http://localhost:50070/**


#### util.NativeCodeLoader Error

64bit centos 또는 ubuntu에서 hadoop을 돌리면 나오는 에러 메세지입니다.
하둡은 32bit로 컴파일되어 있는데 64bit 머신에서 돌려서 나오는 에러 메세지입니다. 
해결 방법은 하둡을 64bit 컴파일 시켜주는 것입니다.

정확하게는 Error는 아니고 Warning이기 때문에 이 메세지가 출력되도 상관은 없지만, 저처럼 메세지가 계속 나와서 짜증난다면 
다음의 지침대로 해결해줄수 있습니다.

{% highlight bash %}
WARN  [main] util.NativeCodeLoader (NativeCodeLoader.java:<clinit>(62)) - Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
{% endhighlight %}

다음을 설치해줍니다.

{% highlight bash %}
sudo apt-get install libprotobuf-dev protobuf-compiler maven libssl-dev build-essential pkgconf cmake
{% endhighlight %}

이때 protobuf 는 반드시 2.5 여야 합니다. <br>
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

다운로드 페이지에서 hadoop-2.7.1-src.tar.gz 파일을 다운로드 합니다. (소스 코드 파일)
압축을 해제시키고 압축을 해제한 폴더로 들어갑니다.

* package 는  build 명령어
* -Pdist는 native code, documentation 없이 build하라는 뜻
* -Pdist,native 는 native code와 함께 build하라는 뜻
* -Pdist,native,docs 는 native code 그리고 documentation과 함께 빌드하라는 뜻
* -DskipTests 테스트를 skip
* -Dtar 는 tar파일을 만듭니다.

**hduser로 로그인해서 해야합니다. 반드시!**

{% highlight bash %}
tar xvf hadoop-2.7.1-src.tar.gz
cd hadoop-2.7.1-src
mvn package -Pdist,native -DskipTests -Dtar
sudo cp -R hadoop-dist/target/hadoop-2.7.1 /usr/local/

{% endhighlight %}

빌드가 끝난후 hadoop-dist/target/hadoop-2.7.1 디렉토리를 /usr/local 에다가 복사합니다.<br>
또는 이미 설치가 되있는 상태라면, hadoop-dist/target/hadoop-2.7.1/lib/native 안의 내용물만 복사하면 됩니다.

확인 하는 방법..
{% highlight bash %}
hadoop checknative -a
{% endhighlight %}

다른 방법...

{% highlight bash %}
hduser:hadoop>file ./lib/native/libhadoop.so.1.0.0 
./lib/native/libhadoop.so.1.0.0: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, BuildID[sha1]=cb34de90c6ae1192cf393441f9a5e0f6f0465e7c, not stripped
{% endhighlight %}






          
[supported-java]: http://wiki.apache.org/hadoop/HadoopJavaVersions
[hadoop-download]: http://www.apache.org/dyn/closer.cgi
[hdfs-wiki]: http://hadoop.apache.org/docs/r2.4.1/hadoop-project-dist/hadoop-hdfs/hdfs-default.xml

[conf-core-site.xml]: {{ page.asset_path }}core-site.xml
[conf-hdfs-site.xml]: {{ page.asset_path }}hdfs-site.xml
[conf-hadoop-env.sh]: {{ page.asset_path }}hadoop-env.sh

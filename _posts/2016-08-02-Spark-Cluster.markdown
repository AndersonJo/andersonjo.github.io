---
layout: post
title:  "Spark Cluster & Yarn"
date:   2016-08-02 01:00:00
categories: "spark"
asset_path: /assets/posts/Spark-Cluster/
tags: ['hadoop']

---

<div>
    <img src="{{ page.asset_path }}spark.jpg" class="img-responsive img-rounded">
</div>


# YARN (Yet Another Resource Negotiator)

### Limitations of classical MapReduce

기존 MapReduce 방식은 scalability, resource utilization, 다른 형태의 데이터 프로세싱(MapReduce이외의) 같은 문제점들이 있었습니다.

- JobTracker (a single master process) 가 모든 jobs들을 관리(coordinate)합니다.
- TaskTrackers (a number of subordinate processes)는 주어진 task를 실행시키며, 주기적으로 JobTracker에 프로그래스를 보고합니다.

<img src="{{ page.asset_path }}mr1.png" class="img-responsive img-rounded">

즉 대규모 cluster에서 나타날수 있는 문제점은 단 하나의 Job Tracker가 모든 jobs들을 관리하기 때문에 bottleneck이 생길수 있습니다.
야후에 따르면 5000개의 nodes 그리고 40,000개의 tasks들을 동시(concurrently)처리가 한계점이라고 합니다.
또한 작거나 너무 큰 하둡 클러스터는 computational resources를 효율적으로 사용하지 못했습니다. 또한 MapReduce jobs이외에는 다른
데이터 프로세싱을 돌릴수가 없었습니다. 이에 2010년 야후 엔지니어들을 하둡의 새로운 아키텍쳐를 만들기 시작했습니다.


### Addressing the scalability issue

<img src="{{ page.asset_path }}busy-jobtracker.png" class="img-responsive img-rounded">

위와같이 JobTracker혼자서 computational resources과리및, 모든 태스크들의 coordination을 하기 때문에 제약이 생길수 밖에 없는 구조입니다. 

해당 이슈를 해결하는 방법은 간단합니다. 
혼자서 모든 일을 처리하는 JobTracker의 일을 TaskTrackers들에게 나누고, 
JobTracker는 cluster resource management (aka **Global ResourceManager**) 그리고 task coordination(aka **ApplicationManster** - AM) 으로 2개로 역활을 나누는 것입니다. 


### YARN (The next generation of Hadoop's compute platform)

먼저 이전 개념과 YARN에서의 용어를 붙여놓자면.. 다음과 같습니다.

| MR1 | YARN |
|:----|:-----|
| A cluster manager | ResourceManager |
| dedicated and short-lived JobTracker | ApplicationMaster |
| TaskTracker | NodeManager |
| MapReduce Job | A distributed application |

<img src="{{ page.asset_path }}yarn.png" class="img-responsive img-rounded">


**ResourceManager**

- a global ResourceManager는 하나의 master deamon으로 돌아가게 되며, 보통 a dedicated machine위에 돌아가게 됩니다.
- live nodes 그리고 resource가 얼마나 가용가능한지 추적합니다.
- 유저가 보낸 application을 언제 얼마나 리소스를 사용할지 coordinate합니다.
- a single processor입니다. 

**ApplicationMaster**

- 유저가 application을 submit하면, 굉장히 가벼운 프로세스인 "ApplicationMaster가 시작됩니다.
- application을 실행시키는데 필요한 일들을 처리합니다. -> 모니터링, 실패한 task에 대해서 재시작 등등
- 컨테이너 안에서 여러가지 태스크들을 돌릴수 있습니다. 
  예를 들어서 MapReduce ApplicationMaster는 a container에게 a map 또는 a reduce task를 실행하라고 요청할수 있습니다. 
  반면에 Giraph ApplicationMaster는 a container에게 Giraph task를 돌리라고 요청할수 있습니다.
  또한 Custom ApplicationMaster를 만들수도 있습니다. (Apache Twil을 통해서 쉽게 만들수 있음)
  

**NodeManager**

- TaskTracker의 더 효율적인 버젼이라고 생각하면 됩니다.
- 고정된 map, reduce slots을 갖고 있는 대신에, dynamically created resource containers들을 갖고 있습니다.
- 컨테이너 안에는 여러 자원을 갖고 있을수 있습니다. 예를 들어 CPU, disk, network IO등등. 하지만 현재는 memory 그리고 CPU (YARN-3)만 지원이 되고 있습니다. 
 
 
 
### Application Submission in YARN

<img src="{{ page.asset_path }}yarn-app-submission.png" class="img-responsive img-rounded">
 
사용자가 application을 ResourceManager로 hadoop jar 명령어 쳐서 (MRv1 처럼)보내면 다음과 같은 일이 발생하게 됩니다.

**1. Run Application**

- ResourceManger는 클러스터 안에서 실행되고 있는 applications의 리스트와, <br>각각의 NodeManager에서 사용가능한 resource 리스트를 갖고 있습니다.
- ResourceManager는 그 다음 어떤 application이 얼마만큼의 resource를 할당 받아야 할지 결정해야 합니다.<br>
   자원사용량은 Queue capacity, ACLs, fairness등등 많은 constraints에 의해서 결정이 됩니다.
- ResourceManager는 pluggable Scheduler를 사용하며, 스케쥴러는 containers의 형태로 cluster resource를 누가, 언제 받을지를 결정합니다.
- ResourceManager가 새로운 application을 받으면, 스케쥴러는 어느 컨테이너에 ApplicationMaster를 실행시킬지를 선택합니다.

**2. Start AM**<br>
**3. Negotiate Resources**

- ApplicationMaster가 실행되고 난뒤, 해당 application의 전체 life cycle을 책임지게 됩니다.
- 제일먼저 AM은 application의 tasks를 처리하기 위한 containers를 받기 위해서 resource request를 ResourceManager에게 보내게 됩니다.
- a resource request는 applications'의 tasks를 처리하기 위해서 resource requirements를 만족시키는 "컨테이너의 갯수"를 요청하는 것입니다.
- resource requirements는 필요한 메모리(megabytes), CPU, 위치 (hostname, rackname 또는 *등으로 표현), priority 등을 갖고 있습니다.
- ResourceManager는 가능한시점에 resource request를 만족하는 container(container ID 그리고 hostname으로 표현)를 할당합니다.
- 컨테이너는 application이 할당된 자원을 특정 host에서 사용하도록 허용합니다.

**4. Launch tasks in the containers**

- Container가 할당된뒤, ApplicationMaster는 NodeManager(해당 host를 관리하는 놈)에게 해당 리소스를 사용하여 tasks를 실행하도록 요청합니다. (MapReduce, Giraph task 등등)
- NodeManager는 tasks들을 모니터링하지 않습니다. NodeManager는 오직 컨테이너 안에서 자원사용량만 모니터링합니다.
  또한 할당된 자원보다 더 많이 사용하면 container자체를 kill해버립니다.
  
**Complete!**

- ApplicationMaster 계속해서 application을 완료하기 위해 계속해서 tasks를 실행시킬 containers를 negotiate합니다.
  AM은 또한 application 그리고 tasks들의 progress를 모니터링 하며, 실패한 tasks들은 새로 requested된 containers안에서 재시작시키며, 
  클라이언트에게 application의 진행상황을 알려줍니다.
- application이 완료된 이후, ApplicationMaster는 자기자신을 종료시키고, AM이 있었던 container또한 반환을 하게 됩니다.
- ResourceManager는 application안의 어떠한 tasks에 대해서도 모니터링을 하지 않지만, ApplicationMasters들의 health는 체크합니다.
  ApplicationManster가 실패하게되면, ResourceManager에 의해서 새로운 container에서 재시작 될 수 있습니다.
- 즉.. ResourceManager는 ApplicationMasters를 관리하고, ApplicationMasters는 tasks들을 관리한다고 보면 됩니다.


# Spark YARN Cluster

### Overview

YARN에서는 각각의 application instance는 ApplicationMaster를 갖고 있습니다.
AM은 ResourceManager로부터  resource를 요청하며, 자원이 할당되면, NodeManager에게 containers를 할당된 자원으로 실행시킬것을 요청합니다.

<img src="{{ page.asset_path }}cluster_deployment_mode.png" class="img-responsive img-rounded">

Spark Cluster mode에서는, **Spark drive는 ApplictionMaster안에서 실행**이 됩니다.
해당 AM은 application실행과, 자원요청을 담당하게 됩니다.

| Mode | YARN Client Mode | YARN Cluster Mode |
|:-----|:-----------------|:------------------|
| Driver                    | Client            | ApplicationMaster | 
| Requests resources        | ApplicationMaster | ApplicationMaster |
| Starts executor processes | YARN NodeManager   | YARN NodeManager |
| Persistent services       | YARN ResourceManager and NodeManagers | YARN ResourceManager and NodeManagers |
| Supports Spark Shell      | Yes                | No               |



### SparkPi Test on YARN host 

{% highlight bash %}
cd  /usr/hdp/current/spark-client
export HADOOP_USER_NAME=spark
sudo -u spark spark-submit --class org.apache.spark.examples.SparkPi --master yarn --num-executors 3 --driver-memory 512m --executor-memory 512m --executor-cores 1 lib/spark-examples*.jar 10
{% endhighlight %}

### SParkPi Test remotely

먼저 client-side configurations 파일들을 가르키는 HADOOP_CONF_DIR 또는 YARN_CONF_DIR가 필요합니다.<br>
모든 파일이 다 필요한 것은 아니고, **core-site.xml** 그리고 **yarn-site.xml**만 있으면 됩니다.
이렇게 하는 이유는 spark-submit을 할때  --master 옵션에 Standalone Cluster 또는 Mesos와는 다르게 주소가 아닌 yarn이 들어가기 때문입니다.

**Copying core-site.xml and yarn-site.xml to my computer**
 
{% highlight bash %}
mkdir -p ~/apps/hdp_conf
scp -i ~/.ssh/dev.pem ubuntu@yarn-master:/etc/hadoop/2.4.2.0-258/0/yarn-site.xml ~/apps/hdp_conf/
scp -i ~/.ssh/dev.pem ubuntu@yarn-master:/etc/hadoop/2.4.2.0-258/0/core-site.xml ~/apps/hdp_conf/
export HADOOP_CONF_DIR=/home/anderson/apps/hdp_conf/
{% endhighlight %}

{% highlight bash %}
spark-submit --class org.apache.spark.examples.SparkPi --master yarn --num-executors 3 --driver-memory 512m --executor-memory 512m --executor-cores 1 $SPARK_HOME/examples/jars/spark-examples*.jar 10
{% endhighlight %}


# Spark Standalone Cluster on AWS



# Spark Network Configuration

**Standalone mode only**

| From | To | Default Port | Purpose | Configuration Setting | Notes |
|:-----|:---|:-------------|:--------|:----------------------|:------|
| Browser | Standalone Master | 8080 | Web UI | spark.master.ui.port SPARK_MASTER_WEBUI_PORT  | Jetty-based. Standalone mode only. |
| Browser | Standalone Worker | 8081 | Web UI | spark.worker.ui.port SPARK_WORKER_WEBUI_PORT  | Jetty-based. Standalone mode only. |
| Driver / Standalone Worker | Standalone Master | 7077 | Submit job to cluster Join cluster | SPARK_MASTER_PORT | Set to "0" to choose a port randomly. Standalone mode only. |
| Standalone Master | Standalone Worker | (random) | Schedule executors | SPARK_WORKER_PORT | Set to "0" to choose a port randomly. Standalone mode only. |

**All cluster managers**

| From | To | Default Port | Purpose | Configuration Setting | Notes |
| Browser | Application | 4040 | Web UI | spark.ui.port | Jetty-based |
| Browser | History Server | 18080 | Web UI | spark.history.ui.port | Jetty-based |
| Executor / Standalone Master | Driver | (random) | Connect to application Notify executor state changes | spark.driver.port | Set to "0" to choose a port randomly. |
| Executor / Driver | Executor / Driver | (random) | Block Manager port | spark.blockManager.port | Raw socket via ServerSocketChannel |



### References

- [IBM Yarn Intro][IBM Yarn Intro]

[IBM Yarn Intro]: http://www.ibm.com/developerworks/library/bd-yarn-intro/
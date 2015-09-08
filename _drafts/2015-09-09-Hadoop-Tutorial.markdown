---
layout: post
title:  "Hadoop Tutorial"
date:   2015-09-09 01:00:00
categories: "hadoop"
asset_path: /assets/posts/Hadoop-Tutorial/
---
<div>
    <img src="{{ page.asset_path }}server.jpg" class="img-responsive img-rounded">
</div>

하둡 설치방법은 [Installing Hadoop on Ubuntu][installing-hadoop]을 참고 해주시기 바랍니다.

[installing-hadoop]: /hadoop/2015/09/08/Installing-Hadoop/

## Starting HDFS

물론.. start-dfs.sh 파일을 실행시킬수도 있지만.. 에러가 날수도 있습니다.
이유는 namenode 와 datanode 서버가 실행될때.. 뭔가 준비가 안된 상태에서 2개를 동시에 열다 보니.. 생겨나는 에러인듯 합니다.
start-dfs.sh 로 실행시켜줘도 되지만.. 만약 tutorial중에 에러가 난다면 다음처럼 각각 따로 따로 하둡을 실행시켜주는게 좋습니다.

각각의 터미널창 열고 실행하시면 됩니다.

{% highlight bash %}
hdfs namenode
hdfs datanode
{% endhighlight %}

## Interacting with HDFS 

여러가지 명령어들이 있는데 대표적인 명령어들만 배워봅시다 :)

* [HDFS Commands][hdfs-commands]

#### Listing files 

dfs는 File System 의 약자

{% highlight bash %}
hdfs dfs -ls /
{% endhighlight %}

#### Make Directories

{% highlight bash %}
hdfs dfs -mkdir -p /user/anderson
hdfs dfs -find /
# /
# /user
# /user/anderson
{% endhighlight %}


#### Upload a file

MongoDB의 GridFS와 유사합니다.<br>
었재든.. 뭐.. 재미있는 부분은 만약에 디렉토리가 존재하지 않을때 (HDFS안에..) 자동으로 만들어주는데..
예를 들어서 /haha/ 에 카피한다고 하면 /haha 디렉토리가 만들어집니다.
문제는 /a/b 같이 missing directories 가 연속으로 있으면 fail로 떨어집니다.

{% highlight bash %}
hdfs dfs -put data.jpg  /user/anderson/
hdfs dfs -ls /user/anderson
# Found 1 items
# -rw-r--r--   3 hduser supergroup  737508 2015-09-09 05:13 /user/anderson/data.jpg
{% endhighlight %}

#### Retrieve Data from HDFS

여러가지 방법들이 있지만 cat을 바로 출력시킬수도 있습니다.

{% highlight bash %}
hdfs dfs -cat /user/anderson/data.jpg
{% endhighlight %}







[hdfs-commands]: http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HDFSCommands.html
---
layout: post
title:  "Hadoop Tutorial"
date:   2015-09-09 01:00:00
categories: "hadoop"
tags: ['hdfs', 'upload file', 'dfsadmin', 'mapreduce']
asset_path: /assets/posts/Hadoop-Tutorial/
---
<div>
    <img src="{{ page.asset_path }}server.jpg" class="img-responsive img-rounded">
</div>

하둡 설치방법은 [Installing Hadoop on Ubuntu][installing-hadoop]을 참고 해주시기 바랍니다.

[installing-hadoop]: /hadoop/2015/09/08/Installing-Hadoop/

## Starting HDFS

start-dfs.sh 로 실행시켜주면 되지만, 만약 에러또는 디버깅이 필요할경우 Daemon으로 띄우지 말고, 집접 실행시키는 방법도 있습니다.

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

put 의 반대 get

{% highlight bash %}
hdfs dfs -get /user/anderson/data.jpg
{% endhighlight %}


## DFSADMIN

dfs 는 각각의 파일을 관리하는데 사용된다면 dfsadmin은 시스템 전체적인 관리에 주로 사용됩니다.

{% highlight bash %}
hdfs dfsadmin -report
{% endhighlight %}


## MapReduce in Python

* [Download techcrunch.csv][techcrunch.csv]
* [Download mapper.py][mapper.py]
* [Download reducer.py][reducer.py]

techcrunch.csv는 미국에서 투자받은 회사정보 입니다.<br>
알고 싶은것은 회사당 총 투자금액이 얼마가 되는지 입니다. 

먼저 hdfs 에 하둡에서 필요한 데이터를 올려줍니다.
{% highlight bash %}
hdfs dfs -mkdir /techcrunch
hdfs dfs -put techcrunch.csv /techcrunch
{% endhighlight %}

그 다음으로 Python으로 mapper 그리고 reducer 를 만들어줄것인데, 매번 디버깅 할때마다 
hdfs에서 가져와서 하면 작업의 효율성이 떨어지니 다음과 같은 방식으로 디버깅을 합니다.

{% highlight bash %}
cat techcrunch.csv | ./mapper.py | sort -k1,1 | ./reducer.py
{% endhighlight %}


#### mapper.py

{% highlight python %}
#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys

last_turf = None
turf_count = 0

for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.split(',')
    company = line[1].strip()
    raise_amt = line[7].strip()

    print '%s,%s' % (company, raise_amt)

{% endhighlight %}



#### reducer.py

{% highlight python %}
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys

current_company = None
total_amt = 0

# input comes from STDIN
for line in sys.stdin:
    company, raise_amt = line.split(',', 1)

    try:
        raise_amt = int(raise_amt)
    except ValueError:
        continue

    if current_company == company:
        total_amt += raise_amt
    else:
        if current_company:
            print '%s\t%s' % (current_company, total_amt)
        total_amt = raise_amt
        current_company = company

if current_company == company:
    print '%s\t%s' % (current_company, total_amt)

{% endhighlight %}


#### Running Map Reduce

Java기반의 MR이 아닌 다른 언어(여기에서는 Python)으로 할때는 Streaming으로 데이터를 보내서 분석을 하게 됩니다.
코드에서 나왔듯이 한줄 한줄씩 처리를 하게 됩니다. 
streaming을 사용하기 위해서는 /usr/local/hadoop-2.7.2/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar 파일을 사용합니다.

{% highlight bash %}
hadoop jar hadoop-streaming-2.7.1.jar -mapper ./mapper.py  -reducer ./reducer.py  -input /techcrunch.csv -output /output

hdfs dfs -cat /output/part-0000023andMe	9000000
# 3Jam	4000000
# 4HomeMedia	2850000
# 5min	5300000
# 750 Industries	1000000
# ...
{% endhighlight %}




# MapReduce in Java

### gradle.build

{% highlight bash %}
group 'io.andersonjo.techcrunch'
version '1.0'
description = '앤더슨의 하둡 맵레듀스 테스트'

apply plugin: 'java'

sourceCompatibility = 1.8

repositories {
    mavenCentral()
}

dependencies {
    compile 'org.apache.hadoop:hadoop-client:2.2.0'
    testCompile 'junit:junit:4.11'
}
{% endhighlight %}

### io.andersonjo.techcrunch

Techcrunch.java

{% highlight java %}
package io.andersonjo.techcrunch;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;

public class Techcrunch {
    public static class TechcrunchMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        public void map(LongWritable key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] data = line.split(",");
            String companyName = data[0];
            int invest = 0;
            try {

                invest = Integer.parseInt(data[7]);
            } catch (NumberFormatException e) {
                invest = 0;
            }

            System.out.println(companyName + " " + invest);
            context.write(new Text(companyName), new IntWritable(invest));
        }
    }

    public static class TechcrunchReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context) 
               throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable investment : values) {
                sum += investment.get();
            }

            context.write(key, new IntWritable(sum));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf);
        job.setJarByClass(Techcrunch.class);

        // Output
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // Map & Reduce Class
        job.setMapperClass(TechcrunchMapper.class);
        job.setReducerClass(TechcrunchReducer.class);

        // I/O Format
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        boolean status = job.waitForCompletion(true);
        if (status) {
            System.exit(0);
        } else {
            System.exit(1);
        }
    }
}

{% endhighlight %}


### Run!

{% highlight bash %}
gradle clean
gradle build
hadoop jar techcrunch-1.0.jar io.andersonjo.techcrunch.Techcrunch /techcrunch.csv /out
{% endhighlight %}

결과는 다음과 같이 나옵니다. 

{% highlight text %}
23andme	9000000
3jam	4000000
4homemedia	2850000
5min	5300000
750-industries	1000000
a123systems	100000000
accertify	4000000
accountnow	12750000
acinion	21000000
acquia	7000000
{% endhighlight %}


[hdfs-commands]: http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HDFSCommands.html
[techcrunch.csv]: {{ page.asset_path }}techcrunch.csv
[mapper.py]: {{ page.asset_path }}mapper.py
[reducer.py]: {{ page.asset_path }}reducer.py
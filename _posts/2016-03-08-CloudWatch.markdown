---
layout: post
title:  "AWS Useful Tips"
date:   2016-03-08 01:00:00
categories: "aws"
static: /assets/posts/AWS/
tags: ['amazon', '아마존', 'cloudwatch']
---


# CloudWatch

<img src="{{ page.static }}cloudwatch.png" class="img-responsive img-rounded">

### More Detailed Mornitoring

* http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/mon-scripts.html

memory, swap, and disk space 등등 추가적인 CloudWatch Metrics를 생성하기 위해서는 다음과 같이 설정합니다.

{% highlight bash %}
sudo apt-get update
sudo apt-get install unzip
sudo apt-get install libwww-perl libdatetime-perl

curl http://aws-cloudwatch.s3.amazonaws.com/downloads/CloudWatchMonitoringScripts-1.2.1.zip -O
unzip CloudWatchMonitoringScripts-1.2.1.zip
rm CloudWatchMonitoringScripts-1.2.1.zip
cd aws-scripts-mon
{% endhighlight %}

aws configure 해서 IAM 유저 설정해주고, 해당 유저는 다음의 권한을 갖고 있어야 합니다.<br>
그뒤 **awscreds.template** 파일을 설정함으로서 IAM Role을 설정할수 있습니다.<br>
이때 중요한점은 **awscreds.conf를 만들어야 한다는 것**입니다.

{% highlight bash %}
cloudwatch:PutMetricData
cloudwatch:GetMetricStatistics
cloudwatch:ListMetrics
ec2:DescribeTags
{% endhighlight %}

테스트는 다음과 같이 할수 있습니다.
{% highlight bash %}
./mon-put-instance-data.pl --mem-used --mem-used --mem-avail --memory-units=megabytes --disk-path=/ --disk-space-util --disk-space-used --disk-space-avail --disk-space-units=megabytes --verify
{% endhighlight %}

작동을 잘 하면, Crontab에다가 등록시켜주면 됩니다.<br>
crontab에다 등록시 --from-cron 옵션을 주어야 합니다.<br>
이때! **--verify는 반드시 삭제**해주어야 합니다.

{% highlight bash %}
* * * * * ~/aws-scripts-mon/mon-put-instance-data.pl --mem-used --mem-used --mem-avail --memory-units=megabytes --disk-path=/ --disk-space-util --disk-space-used --disk-space-avail --disk-space-units=megabytes --from-cron
{% endhighlight %}



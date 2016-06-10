---
layout: post
title:  "AWS Useful Tips"
date:   2016-03-08 01:00:00
categories: "aws"
static: /assets/posts/AWS/
tags: ['amazon', '아마존', 'CloudWatch', 'API Gateway', 'Lambda']

---


# CloudWatch

<img src="{{ page.static }}cloudwatch.png" class="img-responsive img-rounded">

### More Detailed Mornitoring

* [mon-script][mon-script]

memory, swap, and disk space 등등 추가적인 CloudWatch Metrics를 생성하기 위해서는 다음과 같이 설정합니다.

{% highlight bash %}
sudo apt-get update
sudo apt-get install unzip
sudo apt-get install libwww-perl libdatetime-perl

curl http://aws-cloudwatch.s3.amazonaws.com/downloads/CloudWatchMonitoringScripts-1.2.1.zip -O
unzip CloudWatchMonitoringScripts-1.2.1.zip
rm CloudWatchMonitoringScripts-1.2.1.zip
cd aws-scripts-mon
mv awscreds.template awscreds.conf
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


[mon-script]: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/mon-scripts.html


# Authorizer with API Gateway & Lambda 

### Custom Authorizer Architecture

<img src="{{ page.static }}custom-auth-workflow.png" class="img-responsive img-rounded">

OAuth, SAML 등 다양한 authentication 방법들을 API Gateway의 custom authorization을 통해서 컨트롤 할 수 있습니다.
Client가 request를 API Gateway로 authorization token을 헤더로 포함해서 보내면, 해당 Request를 Lambda로 보내고, 
Lambda는 다시 IAM Policies를 리턴시켜서 보냅니다. Policy가 유효하지 않거나, Denied가 되면, 해당 API에대한 call은 실패하게 됩니다.
Valid Policy를 보낸다면, API Gateway는 returned policy를 캐쉬시키고, 동일한 토큰을 갖고서 요청하는 모든 requests를 
미리 설정된 TTL(기본값 3600초)값 동안 Lambda호출없이 처리하게 됩니다.
(현재 Maximum TTL은 3600초이며, 그 이상 넘어갈수 없으며, 0초로 만들어서 캐쉬를 없앨수도 있습니다.)

### Create Custom Authorizer Lambda Function

새로 만드는 Lambda Function이 AWS의 다른 서비스를 호출한다면, 먼저 execution role 설정을 통해서 권한 부여가 필요합니다.


#### **API Gateway -> Lambda**

{% highlight json %}
{
    "type":"TOKEN",
    "authorizationToken":"<caller-supplied-token>",
    "methodArn":"arn:aws:execute-api:<regionId>:<accountId>:<apiId>/<stage>/<method>/<resourcePath>"
}
{% endhighlight %}

| Name | Description |
|:-----|:------------|
| authorizationToken | 클라이언트가 Api Gateway로 request의 header에 붙여서 보내는 Auth-Token의 개념 |
| type | payload type을 정의하며, 현재의 유일한 유효한 값은 "TOKEN" literal 하나입니다. |
| methodArn | API Gateway가 Lambda function에 값을 보내기전에 자동으로 넣어서 보냅니다. |

#### **Authorizer function in Lambda -> API Gateway**

Customer authorizer's Lambda function은 반드시 principal identifier 그리고 policy document를 포함한 response를 리턴시켜야 합니다.

{% highlight json %}

{
  "principalId": "xxxxxxxx",
  "policyDocument": {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": "execute-api:Invoke",
        "Effect": "Allow|Deny",
        "Resource": "arn:aws:execute-api:<regionId>:<accountId>:<appId>/<stage>/<httpVerb>/[<resource>/<httpVerb>/[...]]"
      }
    ]
  }
}

// Example
// GET Method를 Deny 시키는 예제 입니다.
{
  "principalId": "user",
  "policyDocument": {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": "execute-api:Invoke",
        "Effect": "Deny",
        "Resource": "arn:aws:execute-api:us-west-2:123456789012:ymy8tbxw7b/*/GET/"
      }
    ]
  }
}
{% endhighlight %}

| Name | Description |
| Effect | Allow, Deny로 해당 API Gateway의 Action을 실행시킬지 말지 결정합니다. |
| Action | Action은 Resource를 정의하는 API Gateway Execution Service입니다. |
| Resource | * (wild card)를 사용해서 Resource 를 정의할수 있습니다. |
| PrincipalId | $context.authorizer.principalId 변수를 사용해 mapping table 에 access할 수 있습니다. |

#### Create a Custom Authorization for API Methods

<img src="{{ page.static }}gateway_authorizer.png" class="img-responsive img-rounded">

<img src="{{ page.static }}custom-auth-set-authorizer-on-method.png" class="img-responsive img-rounded">
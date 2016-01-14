---
layout: post
title:  "Push Notification via AWS"
date:   2015-12-25 01:00:00
categories: "AWS"
asset_path: /assets/posts/Push-Notification/
tags: ['Celery', 'RabbitMQ', 'GCM']
---
# Google GCM API

Google API Console -> GCM -> Create Credentials

<img src="{{ page.asset_path }}google01.png" class="img-responsive img-rounded" style="border:1px solid #aaa;">

<img src="{{ page.asset_path }}google02.png" class="img-responsive img-rounded" style="border:1px solid #aaa;">











# Create a testing device

### Set up Android

간단하게 Android에 앱 설치하고 GCM에서 메세지 받아볼려면 다음의 tutorial대로 따라하면 손쉽게 안드로이드 설치 가능합니다.

[https://developers.google.com/cloud-messaging/android/start?configured][android-google-gcm]

GCMSender.java 를 API Key로 바꿔줍니다.

{% highlight java %}
public static final String API_KEY = "AIzaSyB2UGIEf-3dVZvtT0CSXzTmrLhGCsMd1XE";
{% endhighlight %}

### Send a message via Topic using Python
Python 에서 다음과 같이 테스트합니다.

{% highlight python %}
from gcm import GCM

gcm = GCM('AIzaSyB2UGIEf-3dVZvtT0CSXzTmrLhGCsMd1XE')
data = {'message': u'아만다 아만다'}

# Topic Messaging
topic = 'global'
gcm.send_topic_message(topic=topic, data=data)
{% endhighlight %}


### Send a message via Token using Python

RegistrationIntentService.java 파일안에 Token 값을 얻어오는 부분에서 추출하면 됩니다.

{% highlight java %}
InstanceID instanceID = InstanceID.getInstance(this);
String token = instanceID.getToken(getString(R.string.gcm_defaultSenderId),
    GoogleCloudMessaging.INSTANCE_ID_SCOPE, null);
{% endhighlight %}


{% highlight python %}
from gcm import GCM

gcm = GCM('AIzaSyB2UGIEf-3dVZvtT0CSXzTmrLhGCsMd1XE')
data = {'message': 'hihihi'}

# Downstream message using JSON request
reg_ids = ['eQRPiEhsH4w:APA91bHY1BpRKeXMCgS1Vr1CphgIbMvuVezjSIY1WwJf9l2AsFvqcUzV55C9drEVg1eSvBHCA8zwHkpxlP2zG8YG5umpIrFunkclTcJNp6Euzv49iIttwRbBmAAwUNICN9HRSgVazXoy']
response = gcm.json_request(registration_ids=reg_ids, data=data)
{% endhighlight %}







# AWS & Python

### Send an Email

* AWS에서 Topics를 만듭니다.
* Create Subscription 버튼을 누르고 자신의 이메일을 넣으면 끝.

{% highlight python %}
import boto3

arn = 'arn:aws:sns:ap-northeast-1:353716070267:test'
sns = boto3.client('sns')

sns.publish(
        TopicArn=arn,
        Subject=u'타이틀 입니다.',
        Message='Python Life is short you need Python'
)
{% endhighlight %}


### Send a message via Endpoint

디바이스에서 GCM에 등록할때 받는 코드.. 즉 endpoint를 활용한 메세지 보내기 방법

{% highlight python %}
endpoint = 'arn:aws:sns:ap-northeast-1:353716070267:endpoint/GCM/fission-test/b25c3d77-75c4-376d-a05f-98786b5eee2b'
sns = boto3.resource('sns')
platform_endpoint = sns.PlatformEndpoint(endpoint)

platform_endpoint.publish(Message='hi')
{% endhighlight %}


### Send a message via Application

{% highlight python %}
arn = 'arn:aws:sns:ap-northeast-1:353716070267:app/GCM/fission-test'
sns = boto3.resource('sns')

platform_app = sns.PlatformApplication(arn)

for endpoint in platform_app.endpoints.all():
    print endpoint.arn

    message = r'''{
        "GCM": "{\"data\":{\"message\":\"Hello World!\",\"url\":\"www.amazon.com\"}}",
        "APNS": "{\"aps\":{\"alert\": \"Hello World!\"} }"
    }'''

    print message
    endpoint.publish(Message=message,
                     MessageStructure='json'
                     )
{% endhighlight %}


[android-google-gcm]:https://developers.google.com/cloud-messaging/gcm
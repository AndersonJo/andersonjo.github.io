---
layout: post
title:  "Push Notification via AWS"
date:   2015-12-25 01:00:00
categories: "AWS"
asset_path: /assets/posts/Push-Notification/
tags: ['Celery', 'RabbitMQ', 'GCM', 'APNS', 'IOS']
---
# Google GCM API

Google API Console -> GCM -> Create Credentials

<img src="{{ page.asset_path }}google01.png" class="img-responsive img-rounded img-fluid" style="border:1px solid #aaa;">

<img src="{{ page.asset_path }}google02.png" class="img-responsive img-rounded img-fluid" style="border:1px solid #aaa;">



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


# IOS APNS

### Configuring SNS

따로 OpenSSL 작업할 필요 없이.. SNS에서 다 해줍니다. <br>
필요한 것은 p12 파일을 Apple에서 받은 다음에 올려주면은 자동으로 Certificate 하고 Private Key를 웹상에서 생성해줍니다.

<img src="{{ page.asset_path }}apns.png" class="img-responsive img-rounded img-fluid" style="border:1px solid #aaa;">






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


### SNS Console

예를 들어서 Android Token을 등록하고 싶다면, Google GCM으로 Application을 만듭니다.

<img src="{{ page.asset_path }}aws01.png" class="img-responsive img-rounded img-fluid" style="border:1px solid #aaa;">

Token은 Android Device에서 GCM으로부터 받은 Token값을 넣어주면 Endpoint ARN은 자동으로 생성됩니다.<br>
여기서 나온 ARN은 카피하고.. Topics로 가서 새로운 Topic을 만듭니다.

<img src="{{ page.asset_path }}aws02.png" class="img-responsive img-rounded img-fluid" style="border:1px solid #aaa;">

새로운 Topic에서 create subscription버튼을 누릅니다.

Protocol은 Application으로 하고, Endpoint에는 카피한 ARN을 넣습니다.

<img src="{{ page.asset_path }}aws03.png" class="img-responsive img-rounded img-fluid" style="border:1px solid #aaa;">

즉..  Topic subscription --> Application ARN --> Device Token 이런 식으로 서로 참조를 하는 구조입니다.


# Using Celery


{% highlight python %}
@shared_task
def push_via_endpoint(endpoint, message):
    """
    Send a push message to an endpoint
    """
    sns = boto3.resource('sns')
    platform_endpoint = sns.PlatformEndpoint(endpoint)
    message_format = r'''{
        "default": "Content",
        "email": "Content",
        "sqs": "Content",
        "lambda": "Content",
        "http": "Content",
        "https": "Content",
        "sms": "Content",
        "APNS": "{\"aps\":{\"alert\": \"Content\"} }",
        "APNS_SANDBOX":"{\"aps\":{\"alert\":\"Content\"}}",
        "APNS_VOIP":"{\"aps\":{\"alert\":\"Content\"}}",
        "APNS_VOIP_SANDBOX": "{\"aps\":{\"alert\": \"Content\"} }",
        "MACOS":"{\"aps\":{\"alert\":\"Content\"}}",
        "MACOS_SANDBOX": "{\"aps\":{\"alert\": \"Content\"} }",
        "GCM": "{ \"data\": { \"message\": \"Content\" } }",
        "ADM": "{ \"data\": { \"message\": \"Content\" } }",
        "BAIDU": "{\"title\":\"Content\",\"description\":\"Content\"}",
        "MPNS" : "<?xml version=\"1.0\" encoding=\"utf-8\"?><wp:Notification xmlns:wp=\"WPNotification\"><wp:Tile><wp:Count>ENTER COUNT</wp:Count><wp:Title>Content</wp:Title></wp:Tile></wp:Notification>",
        "WNS" : "<badge version\"1\" value\"23\"/>"
        }'''
    send_message = message_format % (message, message)
    platform_endpoint.publish(Message=send_message,
                              MessageStructure='json')


@shared_task
def push_via_topic(topic_arn, message):
    print topic_arn, message
    message_format = r'''{
        "default": "Content",
        "email": "Content",
        "sqs": "Content",
        "lambda": "Content",
        "http": "Content",
        "https": "Content",
        "sms": "Content",
        "APNS": "{\"aps\":{\"alert\": \"Content\"} }",
        "APNS_SANDBOX":"{\"aps\":{\"alert\":\"Content\"}}",
        "APNS_VOIP":"{\"aps\":{\"alert\":\"Content\"}}",
        "APNS_VOIP_SANDBOX": "{\"aps\":{\"alert\": \"Content\"} }",
        "MACOS":"{\"aps\":{\"alert\":\"Content\"}}",
        "MACOS_SANDBOX": "{\"aps\":{\"alert\": \"Content\"} }",
        "GCM": "{ \"data\": { \"message\": \"Content\" } }",
        "ADM": "{ \"data\": { \"message\": \"Content\" } }",
        "BAIDU": "{\"title\":\"Content\",\"description\":\"Content\"}",
        "MPNS" : "<?xml version=\"1.0\" encoding=\"utf-8\"?><wp:Notification xmlns:wp=\"WPNotification\"><wp:Tile><wp:Count>ENTER COUNT</wp:Count><wp:Title>Content</wp:Title></wp:Tile></wp:Notification>",
        "WNS" : "<badge version\"1\" value\"23\"/>"
        }'''
    message = message_format.replace('Content', message)
    sns = boto3.resource('sns')
    topic = sns.Topic(topic_arn)
    topic.publish(Message=message, MessageStructure='json')


@shared_task
def sns_create_endpoint(token, data=''):
    sns = boto3.resource('sns')
    platform_application = sns.PlatformApplication(settings.ARN_GCM)
    try:
        return platform_application.create_platform_endpoint(
                Token=token,
                CustomUserData=data
        ).arn
    except ClientError as e:
        print e


@shared_task
def sns_subscribe_topic(topic_arn, endpoint):
    sns = boto3.resource('sns')
    topic = sns.Topic(topic_arn)
    return topic.subscribe(Protocol='application', Endpoint=endpoint)


{% endhighlight %}


[android-google-gcm]:https://developers.google.com/cloud-messaging/gcm
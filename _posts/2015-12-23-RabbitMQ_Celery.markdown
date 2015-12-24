---
layout: post
title:  "RabbitMQ 101"
date:   2015-12-23 01:00:00
categories: "queue"
asset_path: /assets/posts/RabbitMQ_Celery/
tags: []
---
<div>
    <img src="{{ page.asset_path }}bunny.jpg" class="img-responsive img-rounded">
</div>


# RabbitMQ

#### Installation

[Docker RabbitMQ][https://hub.docker.com/_/rabbitmq/]


#### Port

기본 포트는 5672

| Name | Port | ETC  |
|:-----|:-----|:-----|
| epmd | 4369 | 
| Erlang distribution | 25672 |
| AMQP without and with TLS | 5672, 5671 | Python Connectable |
| STOMP if enabled | 61613, 61614 |
| MQTT if enabled | 1883, 8883 |

#### Managing the broker

다음 명령어로 서버를 실행및 중지 가능

{% highlight bash %}
$ rabbitmqctl start
$ rabbitmqctl stop
{% endhighlight %}


#### Default Access 

기본적으로 broker는 guest (암호 guest)라는 유저를 만듭니다.
guest유저는 오직! localhost에서 접속할때만 사용되므로, 다른 machines에서 접속시에는 미리 access control을 할 필요가 있습니다. 

[https://hub.docker.com/_/rabbitmq/]: https://hub.docker.com/_/rabbitmq/


# Hello World

* <a href="{{ page.asset_path }}hello_world.py">hello_world.py</a>

#### Send

{% highlight python %}
import pika

connection = pika.BlockingConnection(
        pika.ConnectionParameters('localdocker', port=32773))
channel = connection.channel()
channel.queue_declare(queue='hello')
{% endhighlight %}

먼저 queue를 만들어야 합니다. queue_declare(queue='hello') 함수를 통해서 hello라는 queue를 만듭니다. <br>
만약 존재하지 않는 queue에 메세지를 보낼경우 RabbitMQ 는 바로 trash시킵니다.

RabbitMQ 에서는 메세지가 바로 queue에 들어가는게 아니라, exchange로 보내게 됩니다.<br>
일단 default exchange는 '' <-- 즉 empty string 입니다.<br>
routing_key에는 queue이름을 적어 줍니다.
 
{% highlight python %}
channel.basic_publish(exchange='', routing_key='hello', body=u'안녕하세요!')
{% endhighlight %}

#### Receive

먼저 현재 RabbitMQ에 어떤 queues 들이 존재한는지 그리고 각각의 queue들에 얼마나 많은 메세지가 쌓여있는지 알고 싶다면 
다음과 같은 명령어를 실행합니다.

{% highlight bash %}
$ rabbitmqctl list_queues
Listing queues ...
hello	1
{% endhighlight %}


받아주는 부분에서도 channel.queue_declare(queue='hello') 를 해줍니다.<br>
이미 보낼때 queue_declare를 해줬는데 다시 해주는 이유는, 실제로는 보내기전에 받는게 먼저 실행이 될 수도 있기 때문입니다.

{% highlight python %}
connection = pika.BlockingConnection(
        pika.ConnectionParameters('localdocker', port=32773))
channel = connection.channel()
channel.queue_declare(queue='hello')
{% endhighlight %}

callback 함수로 메세지를 receive받을수 있습니다.

{% highlight python %}
def callback(ch, method, properties, body):
    print body
    print ch, method, properties, body

channel.basic_consume(callback,
                      queue='hello',
                      no_ack=True)
                      
{% endhighlight %}

start_consuming()을 실행시키면 nerver-ending loop을 통해서 지속적으로 메세지를 받게 됩니다.

{% highlight python %}
channel.start_consuming()
{% endhighlight %}


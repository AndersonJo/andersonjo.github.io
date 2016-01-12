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

#### Docker 

{% highlight bash %}
docker pull rabbitmq
docker run -d -p 5672:5672 --name rabbitmq rabbitmq
{% endhighlight %}

#### Default Access 

기본적으로 broker는 guest (암호 guest)라는 유저를 만듭니다.
guest유저는 오직! localhost에서 접속할때만 사용되므로, 다른 machines에서 접속시에는 미리 access control을 할 필요가 있습니다. 

[https://hub.docker.com/_/rabbitmq/]: https://hub.docker.com/_/rabbitmq/


# Hello World

<img src="{{ page.asset_path }}hello.png" class="img-responsive img-rounded">

* <a href="{{ page.asset_path }}hello_world.py">hello_world.py</a>

### Send

{% highlight python %}
import pika

connection = pika.BlockingConnection(
        pika.ConnectionParameters('172.17.0.1', port=5672))
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

### Receive

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
        pika.ConnectionParameters('172.17.0.1', port=5672))
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












# Work Queue

<img src="{{ page.asset_path }}python-two.png" class="img-responsive img-rounded">

* <a href="{{ page.asset_path }}hello_world.py">new_task.py</a>
* <a href="{{ page.asset_path }}hello_world.py">worker.py</a>

Time-consuming tasks 를 처리하는 방법입니다.<br> 
Worker는 **Round-Robin**방식으로 순차적으로 메세지를 할당받게 되고 task를 처리하게 됩니다.




### Send 

{% highlight python %}
message = raw_input() or "Hello World!"
conn = pika.BlockingConnection(pika.ConnectionParameters('172.17.0.1', 5672))
channel = conn.channel()
channel.queue_declare(queue='task_queue', durable=True)
channel.basic_publish(exchange='',
                      routing_key='task_queue',
                      body=message,
                      properties=pika.BasicProperties(
                          delivery_mode=2,  # make message persistent
                      ))
print(" [x] Sent %r" % message)
{% endhighlight %}

**channel.queue_declare(queue='task_queue', durable=True)**
RabbitMQ가 죽게 되면 모든 queue와 message들은 사라지게 됩니다.<br>
이것을 방지하기 위해서는 durable=True를 해주면 메모리상에 저장하는것이 아닌,<br>
디스크에다가 저장을 하게 됩니다. (물론 받았는데 디스크에 저장전에 다운되면 이 부분까지는 커버안됨)

또한 **delivery_mode = 2** 를 해줌으로서 메세지를 persistent 하게 만들어줍니다.



### Receive (Worker or Consumer)


{% highlight python %}
def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)
    time.sleep(body.count(b'.'))
    print(" [x] Done")
    ch.basic_ack(delivery_tag=method.delivery_tag)


connection = pika.BlockingConnection(
    pika.ConnectionParameters('172.17.0.1', port=5672))
channel = connection.channel()
channel.queue_declare(queue='task_queue', durable=True)

channel.basic_consume(callback, queue='task_queue')
channel.start_consuming()
{% endhighlight %}

**ch.basic_ack(delivery_tag=method.delivery_tag)**<br>
RabbitMQ 는 message를 worker 에 전달하고 바로 메모리에서 삭제를 합니다.<br>
이때 worker 가 task를 실행중에 마저끝내지 못하고 죽었을때는 해당 task 를 날라가 버리며,<br> 
심지어 해당 worker로 전달은 됐지만 아직 처리되지 못한 메세지들도 모두 잃게 됩니다.

worker서버가 죽어도 메세지를 잃어버리지 않게 하기 위해서 message acknowledgments 를 사용합니다.<br>
worker서버에서 message를 처리하고 나면, ack를 RabbitMQ로 보내게 됩니다.<br>
즉 해당 메세지는 모두 처리(processed)되었으니 메모리에서 삭제해도 된다라는 뜻이고,<br> 
RabbitMQ가 ack를 받으면 그때 메모리상에서 메세지를 삭제 시킵니다.

만약 worker(consumer)가 ack를 보내기 전에 죽어버리면, RabbitMQ는 해당 메세지를 re-enqueue시킵니다.<br>
re-enqueue의 경우는 오직 worker가 죽었을때만이고, 테스크처리가 아무리 시간상 오래걸려도 상관없습니다.<br>
* Message acknowledgments 는 기본 자동값으로 켜져 있습니다.











# Publish / Subscribe

<img src="{{ page.asset_path }}exchanges.png" class="img-responsive img-rounded">

* <a href="{{ page.asset_path }}e_consumer.py">e_consumer.py</a>
* <a href="{{ page.asset_path }}e_producer.py">e_producer.py</a>

바로 위의 Worker Queue에서는 하나의 message가 하나의 queue로 보내졌습니다.<br>
하지만 Publish/Subscribe 형태는 전혀 다릅니다. 즉 하나의 메세지가 여러개의 consumers 로 보내지게 됩니다.


### Exchange

이전의 예제에서는 message가 direct로 queue에 들어갔습니다.<br>
하지만 실제 RabbitMQ는 이런식으로는 잘 쓰이지 않습니다. <br>
즉 producer는 실제 종종 어떤 queue한테 메세지를 전달할까 보다는, exchange라는 곳에 보내게 됩니다.<br>
Exchange는 메세지를 어떻게, 어디로, 보낼지 알고 있습니다.

그렇다면 메세지를 어떻게 보낼것인가? append시킬것인가 discard시킬것인가.. 즉.. <br>
Exchange는 Rule에 따라서 행동이 달라지게 되는데, rule은 exchange type에 따라서 변경이 됩니다. <br>

Exchange의 Type으로는 **direct, topic, headers and fanout** 이런 것들이 있습니다.

{% highlight python %}
channel.exchange_declare(exchange='logs', type='fanout')
channel.basic_publish(exchange='logs', routing_key='', body=message)
{% endhighlight %}

exchange_declare를 통해서 exchange type을 설정해주고, <br>
basic_publish에서 exchange argument 값을 logs를 설정해서, logs라는 exchange로 메세지가 들어가게끔 합니다.
만약 exchange='' 이렇게 빈 string을 사용하게 되면, exchange의 default값을 사용한다는 뜻입니다.

{% highlight python %}
conn = pika.BlockingConnection(pika.ConnectionParameters('172.17.0.1', 5672))
channel = conn.channel()
channel.exchange_declare('logs', exchange_type='direct')

while True:
    message = raw_input('Message:')
    channel.basic_publish(exchange='logs', routing_key='', body=message)

{% endhighlight %}

### Consumer


**channel.queue_declare()** 이렇게 queue의 이름을 명시하지 않으면 자동으로 이름이 생성이 됩니다.<br>
(예: amq.gen-JzTY20BRgKO-HjmUJj0wLg)<br>
이전 예제에서는 정확하게 어디 queue에 메세지를 넣을지가 중요하기 때문에, queue의 이름이 중요하지만,<br>
여기서는 exchange를 사용하므로, queue의 이름이 어디인지가 중요하지 않습니다.

exclusive=True옵션을 주면은.. consumer가 disconnect됨가 동시에 해당 queue도 자동으로 삭제가 됩니다.

{% highlight python %}
channel.queue_declare(exclusive=True)
channel.queue_bind(queue=queue_name, exchange='logs')
{% endhighlight %}










# Routing

<img src="{{ page.asset_path }}python-four.png" class="img-responsive img-rounded">
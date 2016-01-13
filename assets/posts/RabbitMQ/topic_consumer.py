import sys

import pika

conn = pika.BlockingConnection(pika.ConnectionParameters('172.17.0.1', 5672))
channel = conn.channel()
channel.exchange_declare('logs_animal', exchange_type='topic')
result = channel.queue_declare('', exclusive=True)
queue_name = result.method.queue

# Ex) korea.dogs.red
binding_keys = sys.argv[1:] if len(sys.argv) > 1 else '*.dogs.*'

for binding_key in binding_keys:
    print 'Binding to %s' % binding_key
    channel.queue_bind(queue_name, exchange='logs_animal', routing_key=binding_key)


def callback(ch, method, properties, body):
    print(" [x] %r:%r" % (method.routing_key, body))


channel.basic_consume(callback,
                      queue=queue_name,
                      no_ack=True)

channel.start_consuming()

import sys

import pika

conn = pika.BlockingConnection(pika.ConnectionParameters('172.17.0.1', 5672))
channel = conn.channel()

channel.exchange_declare('direct_logs', exchange_type='direct')
result = channel.queue_declare()
queue_name = result.method.queue

for routing_key in sys.argv[1:]:
    channel.queue_bind(exchange='direct_logs', queue=queue_name, routing_key=routing_key)


def callback(ch, method, properties, body):
    print "[x] %s" % body


channel.basic_consume(callback, queue=queue_name, no_ack=True)
channel.start_consuming()

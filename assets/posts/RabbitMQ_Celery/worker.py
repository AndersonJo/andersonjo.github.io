import time

import pika


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

# -*- coding:utf-8 -*-

import pika


def send():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('172.17.0.1', port=5672))
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    channel.basic_publish(exchange='', routing_key='hello', body=u'안녕하세요!!')
    connection.close()


def receiver():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('172.17.0.1', port=5672))
    channel = connection.channel()
    channel.queue_declare(queue='hello')

    def callback(ch, method, properties, body):
        print body
        print ch, method, properties, body

    channel.basic_consume(callback, queue='hello', no_ack=True)
    channel.start_consuming()

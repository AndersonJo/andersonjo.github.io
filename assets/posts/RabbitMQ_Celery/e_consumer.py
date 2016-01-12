import pika

conn = pika.BlockingConnection(pika.ConnectionParameters('172.17.0.1', 5672))
channel = conn.channel()

# Configure Exchange
# channel.exchange_delete(exchange='logs', if_unused=True)
exchange = channel.exchange_declare('logs', exchange_type='fanout')

# Configure Queue
result = channel.queue_declare(exclusive=True)
queue_name = result.method.queue
channel.queue_bind(queue=queue_name, exchange='logs')
print 'Queue Name:%s' % queue_name


def callback(ch, method, properties, body):
    print(" [x] %r" % body)


channel.basic_consume(callback,
                      queue=queue_name,
                      no_ack=True)

channel.start_consuming()

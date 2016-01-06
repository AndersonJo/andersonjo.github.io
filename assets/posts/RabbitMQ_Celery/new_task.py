import pika

while True:
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
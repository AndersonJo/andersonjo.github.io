import pika

conn = pika.BlockingConnection(pika.ConnectionParameters('172.17.0.1', 5672))
channel = conn.channel()
channel.exchange_delete('logs_animal')
channel.exchange_declare('logs_animal', exchange_type='topic')

routing_key = 'korea.dogs.red'

count = 0
while True:
    message = raw_input('Message:')
    if message == 'change':
        routing_key = raw_input('New Routing Key:')
        continue

    channel.basic_publish('logs_animal', routing_key=routing_key, body=message)
    print '[%d] Published to %s - %s' % (count, routing_key, message)
    count += 1

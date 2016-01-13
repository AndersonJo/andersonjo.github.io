import pika

conn = pika.BlockingConnection(pika.ConnectionParameters('172.17.0.1', 5672))
channel = conn.channel()

# channel.exchange_delete('logs')
channel.exchange_declare('direct_logs', exchange_type='direct')
routing_key = 'red'
while True:
    message = raw_input('Message:')
    if message == 'change':
        routing_key = raw_input('Routing Key:')
    channel.basic_publish('direct_logs', routing_key=routing_key, body=message)
    print 'Message Sent to %s (%s)' % (routing_key, message)

conn.close()

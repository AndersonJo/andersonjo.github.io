import pika

conn = pika.BlockingConnection(pika.ConnectionParameters('172.17.0.1', 5672))
channel = conn.channel()
channel.exchange_declare('logs', exchange_type='fanout')

while True:
    message = raw_input('Message:')
    channel.basic_publish(exchange='logs', routing_key='', body=message)
    if message == 'exit':
        break
conn.close()

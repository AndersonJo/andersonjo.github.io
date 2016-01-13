from celery import Celery

app = Celery('hello', broker='amqp://172.17.0.1:5672')

@app.task
def hello():
    return 'hello world'



from time import sleep

from celery import Celery
app = Celery('tutorial', backend='redis://172.17.0.1:6381/15', broker='amqp://172.17.0.1:5672')


@app.task
def add(x, y):
    sleep(1)
    return x + y


r = add.delay(4, 4)
print r.result
print 'haha'
print r.wait()
print r.result
# -*- coding:utf-8 -*-
from gcm import GCM

gcm = GCM('AIzaSyB2UGIEf-3dVZvtT0CSXzTmrLhGCsMd1XE')
data = {'message': u'아만다 아만다'}

# Topic Messaging
topic = 'global'
gcm.send_topic_message(topic=topic, data=data)

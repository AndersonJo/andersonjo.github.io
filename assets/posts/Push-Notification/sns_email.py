# -*- coding:utf-8 -*-
import json

import boto3

arn = 'arn:aws:sns:ap-northeast-1:353716070267:test'
sns = boto3.client('sns')

sns.publish(
        TopicArn=arn,
        Subject=u'타이틀 입니다.',
        Message=json.dumps({'GCM': {'data': {'message': 'Hi'}}})
)




# response = client.publish(
#     TopicArn='string',
#     TargetArn='string',
#     Message='string',
#     Subject='string',
#     MessageStructure='string',
#     MessageAttributes={
#         'string': {
#             'DataType': 'string',
#             'StringValue': 'string',
#             'BinaryValue': b'bytes'
#         }
#     }
# )

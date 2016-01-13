# -*- coding:utf-8 -*-

import boto3

arn = 'arn:aws:sns:ap-northeast-1:353716070267:app/GCM/fission-test'
endpoint = 'arn:aws:sns:ap-northeast-1:353716070267:endpoint/GCM/fission-test/b25c3d77-75c4-376d-a05f-98786b5eee2b'
sns = boto3.resource('sns')
platform_endpoint = sns.PlatformEndpoint(endpoint)

platform_endpoint.publish(Message='hi')





# response = platform_endpoint.publish(
#     TopicArn='string',
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

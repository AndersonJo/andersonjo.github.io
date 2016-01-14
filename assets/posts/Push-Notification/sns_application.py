# -*- coding:utf-8 -*-

import boto3

arn = 'arn:aws:sns:ap-northeast-1:353716070267:app/GCM/fission-test'
sns = boto3.resource('sns')

platform_app = sns.PlatformApplication(arn)

for endpoint in platform_app.endpoints.all():
    print endpoint.arn

    message = r'''{
        "GCM": "{\"data\":{\"message\":\"Hello World!\",\"url\":\"www.amazon.com\"}}",
        "APNS": "{\"aps\":{\"alert\": \"Hello World!\"} }"
    }'''

    print message
    endpoint.publish(Message=message,
                     MessageStructure='json')

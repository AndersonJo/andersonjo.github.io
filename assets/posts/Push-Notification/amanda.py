# -*- coding:utf-8 -*-

import boto3

arn = 'arn:aws:sns:ap-northeast-1:353716070267:app/GCM/fission-test'
endpoint = 'arn:aws:sns:ap-northeast-1:353716070267:endpoint/GCM/amanda-gcm/742bbd91-05e0-3a0d-87bc-4cdcbff56833'
sns = boto3.resource('sns')
platform_endpoint = sns.PlatformEndpoint(endpoint)
content = u'흥하자'
message = r'''{
        "GCM": "{\"data\":{\"message\":\"%s\",\"url\":\"www.amazon.com\"}}",
        "APNS": "{\"aps\":{\"alert\": \"%s\"} }"
    }'''
message = message % (content, content)
print message
platform_endpoint.publish(Message=message, Subject=u'타이틀', MessageStructure='json')

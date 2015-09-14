#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""
import httplib
import json



path = '/maps/api/geocode/json?address=Oxford%20University,%20uk&sensor=false'

connection = httplib.HTTPConnection('maps.googleapis.com')
connection.request('GET', path)
content = connection.getresponse().read()
content = json.loads(content)
print content['results'][0]['address_components'][0]['long_name']  # University of Oxford





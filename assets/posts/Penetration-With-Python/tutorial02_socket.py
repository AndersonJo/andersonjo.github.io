#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""

import socket

sock = socket.socket()
sock.connect(('maps.googleapis.com', 80))
sock.sendall(
    'GET /maps/api/geocode/json?address=Oxford%20University,%20uk&sensor=false\n'
    'HOST: maps.google.com:80\n'
    'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36\n'
    'Connection: close\n'
    '\n')
reply = sock.recv(4096)
print reply

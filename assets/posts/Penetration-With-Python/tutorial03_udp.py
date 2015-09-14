#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""

import socket

import sys

HOST = '127.0.0.1'
PORT = 50013
MAX = 65535

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if sys.argv[1:] == ['server']:
    sock.bind((HOST, PORT))
    print 'Listening at', sock.getsockname()
    while True:
        data, address = sock.recvfrom(MAX)
        print 'The client at', address, 'said', data
        sock.sendto('%d bytes' % len(data), address)
else:
    print 'Address before sending ', sock.getsockname()
    sock.sendto(raw_input(), (HOST, PORT))
    print 'Address after sending', sock.getsockname()
    data, address = sock.recvfrom(MAX)
    print '[Server %s]: %s'% (address, data)


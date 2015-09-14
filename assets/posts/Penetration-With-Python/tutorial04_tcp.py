#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""

import socket
import sys
import struct

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

HOST = '127.0.0.1'
PORT = 50000
RECEIVE_SIZE = 2


def recv_msg(sock):
    data = []
    received_length = 0
    data_length = struct.unpack('>I', sock.recv(4))[0]
    while received_length < data_length:
        p = sock.recv(RECEIVE_SIZE)
        data.append(p)
        received_length += RECEIVE_SIZE

    response = ''.join(data)
    return response


def send_msg(sock, message):
    message = message.encode('UTF-8')
    length = len(message)
    packed = struct.pack('>I', length)
    sock.sendall(packed + message)


if sys.argv[1:] == ['server']:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(1)

    while True:
        print 'Listening at', sock.getsockname()
        active_socket, sockname = sock.accept()
        print 'New Active Socket:', sockname
        print 'Passive Socket:', active_socket.getsockname(), ', Active Socket:', active_socket.getpeername()
        message = recv_msg(active_socket)
        print 'Server received %s' % message
        return_message = u'사요나라~! 졸려..'
        send_msg(active_socket, return_message)
        active_socket.close()
    print 'Reply sent, socket clossed'
else:
    sock.connect((HOST, PORT))
    print 'Client has been assigned socket name', sock.getsockname()

    text = raw_input('메세지:').decode('UTF-8')
    send_msg(sock, text)
    reply = recv_msg(sock)
    print 'the Server said:', reply
    sock.close()
    print 'Good bye'

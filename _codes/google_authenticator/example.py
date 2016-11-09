import base64
import hashlib
import hmac
import struct
import time


def get_hotp_token(secret, intervals_no):
    key = base64.b32decode(secret, True)
    msg = struct.pack(">Q", intervals_no)
    h = hmac.new(key, msg, hashlib.sha1).digest()
    o = ord(h[19]) & 15
    h = (struct.unpack(">I", h[o:o + 4])[0] & 0x7fffffff) % 1000000
    return h


def get_totp_token(secret):
    return get_hotp_token(secret, intervals_no=int(time.time()) // 30)


if __name__ == '__main__':
    secret = 'MZXW633PN5XW6MZX'

    for i in xrange(1, 10):
        # print i, get_hotp_token(secret, intervals_no=i)
        print i, get_totp_token(secret)



from hashlib import md5
import numpy as np

hashkey = '91633d65575b2d47a7496707a93dc7e4'

brutal_key = np.zeros(32)

v1 = 0 << 24
v2 = 0 << 24
v3 = 0 << 24
v4 = 0 << 24
for i in range(0, len(hashkey), 2):
    k = int(hashkey[i: i + 2], 16)
    if i < 8:
        v1 = (v1 << 8) | k
        print i, hashkey[i: i + 2], bin(int(hashkey[i: i + 2], 16)), bin(v1), v1

print int('91633d65', 16)
print int('575b2d47', 16)

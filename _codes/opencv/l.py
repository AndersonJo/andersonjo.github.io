from lshash import LSHash
import numpy as np

s = LSHash(10, 8)
s.index([1,2,3,4,5,6,7,8])
print s.hash_tables[0].keys()[0]
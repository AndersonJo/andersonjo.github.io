#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys

last_turf = None
turf_count = 0

for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.split(',')
    company = line[1].strip()
    raise_amt = line[7].strip()

    print '%s,%s' % (company, raise_amt)

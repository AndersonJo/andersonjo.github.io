#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""

import sys

current_company = None
total_amt = 0

# input comes from STDIN
for line in sys.stdin:
    company, raise_amt = line.split(',', 1)

    try:
        raise_amt = int(raise_amt)
    except ValueError:
        continue

    if current_company == company:
        total_amt += raise_amt
    else:
        if current_company:
            print '%s\t%s' % (current_company, total_amt)
        total_amt = raise_amt
        current_company = company

# do not forget to output the last word if needed!
if current_company == company:
    print '%s\t%s' % (current_company, total_amt)
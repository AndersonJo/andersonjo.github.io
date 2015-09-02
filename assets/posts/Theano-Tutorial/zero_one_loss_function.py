# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""
from theano import tensor as T
from theano import function
import numpy as np
# Zero-One Loss
x = T.vector('x')
f = function([x], T.argmax(x))

T.neq
x = [500, 100]

print x
print f(x)

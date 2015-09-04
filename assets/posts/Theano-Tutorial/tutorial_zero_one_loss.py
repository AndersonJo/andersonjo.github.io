# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""

from theano import tensor as T
from theano import function

x = T.dvector('x')
f = function([x], T.sum(T.neq(T.argmax(x), 0)))
print f([1, 2, 3, 4, 5])  # 1
print f([5, 4, 3, 2, 1])  # 0

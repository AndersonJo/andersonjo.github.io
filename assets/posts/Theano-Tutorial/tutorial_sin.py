# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""
from theano import tensor as T
from theano import function


def example_sin():
    """
    Example 01 : Sin
    """
    x = T.dscalar('x')
    f = function([x], T.sin(T.deg2rad(x)) ** 2 + T.cos(T.deg2rad(x)) ** 2)
    print f(0)  # 1.0
    print f(30)  # 1.0
    print f(45)  # 1.0
    print f(60)  # 1.0
    print f(90)  # 1.0


if __name__ == '__main__':
    example_sin()

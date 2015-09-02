# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""
from matplotlib import pylab
from theano import tensor as T
from theano import function
import numpy as np
from theano import shared

m = T.dscalar('m')
b = T.dscalar('b')
x = T.dvector('x')
y = T.dvector('y')
error_function = function([x, y, m, b], (y - (m * x + b)) ** 2)

total_error_value = shared(0)
inc = T.iscalar('inc')
total_error = function([inc], total_error_value, updates=[(total_error_value, total_error_value + inc)])



def graph(data):
    pylab.scatter(data[:, 0], data[:, 1])
    pylab.grid()
    pylab.show()


def main():
    data = np.loadtxt(open('data.csv', 'r'), delimiter=',')
    init_m = 0
    init_b = 0
    print error_function(data[:, 0], data[:, 1], init_m, init_b)


if __name__ == '__main__':
    main()

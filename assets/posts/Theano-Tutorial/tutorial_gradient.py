# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""
from matplotlib import pylab
import numpy as np

from theano import tensor as T
from theano import function
from theano import shared


# Define scalars and vectors
m = T.dscalar('m')
b = T.dscalar('b')
x = T.dvector('x')
y = T.dvector('y')
N = T.iscalar('N')
inc = T.iscalar('inc')

# Define Error Function
error_function = function([x, y, m, b], (y - (m * x + b)) ** 2)

# Define b_gradient or y-intercept
b_gradient = function([x, y, m, b, N], T.sum(2. / N * -(y - (m * x + b))))

# Define m_gradient or slope
m_gradient = function([x, y, m, b, N], T.sum(2. / N * -x * (y - (m * x + b))))


def graph(data):
    pylab.scatter(data[:, 0], data[:, 1])
    pylab.grid()
    pylab.show()


def graph_with_line(data, m, b):
    N = len(data)
    line = data[:, 0] * m + b

    pylab.scatter(data[:, 0], data[:, 1])
    pylab.plot(data[:, 0], line)
    pylab.grid()
    pylab.show()


def run_gradient_descent(data, learning_rate, num_iterations):
    _b = 0
    _m = 0
    _N = len(data)
    for i in range(num_iterations):
        _b -= (b_gradient(data[:, 0], data[:, 1], _m, _b, _N) * learning_rate)
        _m -= (m_gradient(data[:, 0], data[:, 1], _m, _b, _N) * learning_rate)
    return [_m, _b]


def main():
    data = np.loadtxt(open('data.csv', 'r'), delimiter=',')
    init_m = 0
    init_b = 0
    learning_rate = 0.0001
    num_iterations = 1000
    # print error_function(data[:, 0], data[:, 1], init_m, init_b)
    m, b = run_gradient_descent(data, learning_rate, num_iterations)
    print error_function(data[:, 0], data[:, 1], m, b)
    graph_with_line(data, m, b)


if __name__ == '__main__':
    main()

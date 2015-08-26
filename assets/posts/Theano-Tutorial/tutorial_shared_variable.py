# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""
from theano import tensor as T
from theano import function
from theano import shared

# Shared Variable

state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state + inc)])
decrementor = function([inc], state, updates=[(state, state - inc)])

accumulator(1)  # return value: 0
print 'state:', state.get_value()  # state: 1

accumulator(3)  # return value: 1
print 'state:', state.get_value()  # state: 4

decrementor(1)  # return value: 4
print 'state:', state.get_value()  # state: 3

accumulator(1)  # return value: 3
state.set_value(10)
print 'state:', state.get_value()  # state: 10


# Given Parameter

foo = T.scalar(dtype=state.dtype)
fn_of_state = state * 2 + inc
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print skip_shared(1, 3)  # 7
print state.get_value()  # 10


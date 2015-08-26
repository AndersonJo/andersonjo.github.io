# -*- coding:utf-8 -*-
"""
@Author: Anderson Jo 조창민
@email: a141890@gmail.com
@homepage: http://andersonjo.github.io/
"""
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams

random_stream = RandomStreams(seed=1234)
r_uniform = random_stream.uniform(size=(2, 2))
r_normal = random_stream.normal(size=(2, 2))

print dir(r_uniform)

f = function([], r_uniform)
g = function([], r_normal, no_default_updates=True)

print f()  # [[ 0.75718064  0.1130526 ] [ 0.00607781  0.8721389 ]]
print f()  # [[ 0.31027075  0.24150866] [ 0.56740797  0.73226671]]

print g()  # [[ 0.42742609  1.74049825] [-0.02169041  1.48401086]]
print g()  # [[ 0.42742609  1.74049825] [-0.02169041  1.48401086]]

random_stream.seed(7777)
print g()  # [[-0.28277921 -0.12554144] [ 1.56899783 -1.10901327]]

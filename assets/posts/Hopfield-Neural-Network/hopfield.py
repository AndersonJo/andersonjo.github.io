import numpy as np

a = np.matrix('0 1 0 1')
contribution = np.apply_along_axis(lambda x: 2 * x - 1, 0, a)
# [[-1  1 -1  1]]

contribution = contribution.T * contribution
# [[ 1 -1  1 -1]
#  [-1  1 -1  1]
#  [ 1 -1  1 -1]
#  [-1  1 -1  1]]

contribution = contribution - np.identity(4)
# [[ 0. -1.  1. -1.]
#  [-1.  0. -1.  1.]
#  [ 1. -1.  0. -1.]
#  [-1.  1. -1.  0.]]

a.T
# [[0]
#  [1]
#  [0]
#  [1]]

answer = contribution.dot(a.T)
# [[-2.]
#  [ 1.]
#  [-2.]
#  [ 1.]]

answer = np.apply_along_axis(lambda x: 1 if x == 1 else 0, 0, answer.T)

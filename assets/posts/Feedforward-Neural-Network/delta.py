#!/usr/bin/env python
# -*- coding:utf-8 -*-

class Hebb(object):
    def __init__(self):
        # Weights
        self.w1 = 1.
        self.w2 = -1.

        # Learning Rate
        self.rate = 1.

        # Epoch Count
        self._epoch_count = 0;

    def process(self):
        for i in range(5):
            self.epoch()

    def epoch(self):
        print '##### Epoch %d ##############' % self._epoch_count
        self.set_pattern(-1., -1.)
        self.set_pattern(-1., 1.)
        self.set_pattern(1., -1.)
        self.set_pattern(1., 1.)
        self._epoch_count += 1

    def set_pattern(self, i1, i2):
        print 'i1=%2d i2=%2d   ' % (i1, i2),
        result = self.recognize(i1, i2)
        print 'result=%5d    ' % result,

        delta = self.train(self.rate, i1, result)
        self.w1 += delta
        print 'delta1=%4d  ' % delta,

        delta = self.train(self.rate, i2, result)
        self.w2 += delta
        print 'delta2=%4d    ' % delta,
        print 'w1=%d   w2=%d'% (self.w1, self.w2),
        print

    def recognize(self, i1, i2):
        return (self.w1 * i1 + self.w2 * i2) * 0.5

    def train(self, rate, input, output):
        return rate * input * output


if __name__ == '__main__':
    h = Hebb()
    h.process()

from abc import ABCMeta, abstractmethod
import numpy as np
from activation import SigmoidActivationFunction


class FeedforwardNetwork(object):
    def __init__(self):
        self._layers = []
        self._input_layer = None
        self._output_layer = None

    def add_layer(self, layer):
        if self._output_layer:
            layer.set_prev(self._output_layer)
            self._output_layer.set_next(layer)

        if not len(self._layers):
            self._input_layer = self._output_layer = layer
        else:
            self._output_layer = layer

        self._layers.append(layer)

    def reset(self):
        [l.reset() for l in self._layers]

    def compute(self, input):
        if self._input_layer.neuro_size != input.size:
            raise Exception('Size Mismatch - input size=%d,   input layer size:%d' %
                            (self._input_layer.neuro_size, input.size))

        results = None
        for layer in self._layers:
            if layer.is_input_layer():
                results = layer.compute(input)
            elif layer.is_hidden_layer():
                results = layer.compute(results)

        return self._output_layer._actives

    def train(self, output, expected, learning_rate=0.5, momemtum=0.5):
        for layer in self._layers:
            layer.clear_error();

        for layer in self._layers[::-1]:
            if layer.is_output_layer():
                accumulated = layer.train(output, expected)
            else:
                accumulated = layer.train()
            layer.learn(accumulated, learning_rate, momemtum)

    def get_error(self):
        return 0.8

    def __str__(self):
        s = ['Feedforward Network']
        [s.append('%d - Layer\n%s' % (i, l.__str__())) for i, l in enumerate(self._layers)]
        return '\n'.join(s)


class FeedLayer(object):
    def __init__(self, neuro_size, activation=SigmoidActivationFunction()):
        self.neuro_size = neuro_size
        self._activation = activation
        self._actives = np.zeros(neuro_size)  # fire
        self._weights = None
        self._prev_layer = None
        self._next_layer = None

        self.errors = None
        self.deltas = None  # Error Deltas
        self.momemtums = None

    def compute(self, input_data=None):
        if self.is_input_layer() and input_data is not None:
            self._actives = input_data.copy()

        input_data = np.matrix(np.append(self._actives, 1.))

        for i in range(self._next_layer.neuro_size):
            col = self._weights[:, i]
            sum = float(input_data * col)
            self._next_layer._actives.put(i, self._activation.activate(sum))
        return self._actives

    def train(self, output=None, expected=None):
        if output is not None and expected is not None:
            self.errors = expected - output
            self.deltas = np.zeros(self.neuro_size)
            for i in range(self.neuro_size):
                self.deltas.put(i, self.errors[i] * self._activation.derivative(self._actives[i]))
            return

        accumulated = np.matrix(np.zeros((self.neuro_size + 1, self._next_layer.neuro_size)))
        for i in range(self._next_layer.neuro_size):
            for j in range(self.neuro_size):
                accumulated[j, i] += self._next_layer.deltas[i] * self._actives[j]
                self.errors.put(j, self.errors[j] + self._weights[j, i] * self._next_layer.deltas[i])

            accumulated[self.neuro_size, i] += self._next_layer.deltas[i]

        self.deltas = np.zeros(self.neuro_size)
        if self.is_hidden_layer():
            for i in range(self.neuro_size):
                self.deltas[i] = self.errors[i] * self._activation.derivative(self._actives[i])
        return accumulated

    def learn(self, accumulated, learning_rate, momemtum):
        if self._next_layer is not None and self.momemtums is None:
            self.momemtums = np.zeros((self.neuro_size + 1, self._next_layer.neuro_size))

        if self._weights is not None:
            m1 = accumulated * learning_rate
            m2 = self.momemtums * momemtum
            self.momemtums = m1 + m2
            self.set_weights(self._weights + self.momemtums)

            # print m1

    def clear_error(self):
        self.errors = np.zeros(self.neuro_size)

    def set_weights(self, matrix):
        self._weights = matrix

    def set_prev(self, layer):
        self._prev_layer = layer

    def set_next(self, next_layer):
        self._next_layer = next_layer
        # add one to provide a threshold value in row 0
        self._weights = np.matrix(np.random.uniform(-1, 1, size=(self.neuro_size + 1, next_layer.neuro_size)))

    def reset(self):
        if self._weights is not None:
            self._weights = np.matrix(np.random.uniform(-1, 1, size=self._weights.shape))

    def is_input_layer(self):
        return self._prev_layer == None

    def is_hidden_layer(self):
        return self._prev_layer != None and self._next_layer != None

    def is_output_layer(self):
        return self._next_layer == None

    def __str__(self):
        s = [super(FeedLayer, self).__str__()]
        s.append('\tNeuro Size:%d' % self.neuro_size)
        s.append('\t Actives:' + str(self._actives).replace('\n', '\n\t'))
        s.append('\t Errors:' + str(self.errors).replace('\n', '\n\t'))
        s.append('\t Deltas:' + str(self.deltas).replace('\n', '\n\t'))
        s.append('\t' + str(self._weights).replace('\n', '\n\t'))
        return '\n'.join(s)


class Backpropagation(object):
    def __init__(self, network, input, expect, learning_rate, momemtum):
        self._network = network
        self._input = input
        self._expect = expect
        self._learning_rate = learning_rate
        self._momemtum = momemtum

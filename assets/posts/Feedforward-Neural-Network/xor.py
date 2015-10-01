import numpy as np

from feedforward import FeedforwardNetwork, FeedLayer


def main():
    network = FeedforwardNetwork()
    network.add_layer(FeedLayer(2))
    network.add_layer(FeedLayer(3))
    network.add_layer(FeedLayer(1))
    network.reset()
    # print network
    # print

    input = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])
    expect = np.array([0, 1, 1, 0])
    learning_rate = 0.7
    momemtum = 0.9

    epoch = 0
    while epoch < 1000 and network.get_error() > 0.001:
        for i in range(len(input)):
            r = network.compute(input[i])
            network.train([r], expect[i], learning_rate, momemtum)
            if epoch%100 == 0:
                print 'Result..', r, expect[i]
        if epoch%100 == 0:
            print 'Epoch %d '% epoch,  '='*50
        epoch += 1

    print network

    # train = Backpropagation(network, input, expect, learning_rate, momemtum)


if __name__ == '__main__':
    main();

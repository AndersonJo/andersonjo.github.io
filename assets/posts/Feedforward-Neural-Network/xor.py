from feedforward import FeedforwardNetwork, FeedLayer


def main():
    network = FeedforwardNetwork()
    network.add_layer(FeedLayer(2))
    network.add_layer(FeedLayer(3))
    network.add_layer(FeedLayer(1))
    network.reset()




if __name__ == '__main__':
    main();
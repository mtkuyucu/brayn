import random
import math


def sigmoid(x):
    if x < -1e4:
        return 1.
    else:
        return 1 / (1 + math.exp(-0.007*x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return (1 - s) * s


class Input:
    output = 1.
    local_gradient = 0.
    inputs = []

    def propagate(self, neurons):
        pass

    def backpropagate(self, neurons):
        pass


class Neuron(Input):
    def __init__(self, inputs=[]):
        self.inputs = [(i, random.uniform(-1, 1)) for i in [0] + inputs]
        self.output = 0.
        self.local_gradient = 0.

    def propagate(self, neurons):
        self.local_field = sum(neurons[i].output*w for i, w in self.inputs)
        self.output = sigmoid(self.local_field)
        self.local_gradient = 0.

    def backpropagate(self, neurons):
        self.local_gradient *= sigmoid_prime(self.local_field)

        self.inputs = [(i, w - self.local_gradient * neurons[i].output)
                       for i, w in self.inputs]

        for i, w in self.inputs:
            neurons[i].local_gradient += w * self.local_gradient


class NeuralNetwork:
    def __init__(self, n_inputs):
        self.neurons = [Input() for _ in range(n_inputs+1)]
        self.last_layer_start = 1

    def add_layer(self, n_nodes):
        last_layer_end = len(self.neurons)
        last_layer = list(range(self.last_layer_start, last_layer_end))
        self.neurons += [Neuron(last_layer) for _ in range(n_nodes)]
        self.last_layer_start = last_layer_end

    def compute(self, inputs):
        # set inputs
        for idx, value in enumerate(inputs):
            self.neurons[idx+1].output = value
        # propagate
        for neuron in self.neurons:
            neuron.propagate(self.neurons)
        # get output
        last_layer = list(range(self.last_layer_start, len(self.neurons)))
        return [self.neurons[idx].output for idx in last_layer]

    def train(self, inputs, expect):
        # compute actual output
        output = self.compute(inputs)
        # initialiaze gradient
        for i in range(len(output)):
            neuron = self.neurons[self.last_layer_start + i]
            neuron.local_gradient = output[i] - expect[i]
        # backpropagate
        for neuron in reversed(self.neurons):
            neuron.backpropagate(self.neurons)


if __name__ == "__main__":
    nn = NeuralNetwork(2)
    nn.add_layer(20)
    nn.add_layer(1)

    f = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    for _ in range(10000):
        inputs, expect = random.choice(f)
        nn.train(inputs, expect)

    for inputs, expect in f:
        output = nn.compute(inputs)
        print(expect, output)

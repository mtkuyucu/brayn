import random
import math


def sigmoid(x):
    if x < -500:
        return 1.
    else:
        return 1 / (1 + math.exp(-x))


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
        self.last_layer = range(1, 1+n_inputs)

    def add_layer(self, n_nodes):
        layer_start = len(self.neurons)
        self.neurons += [Neuron(list(self.last_layer)) for _ in range(n_nodes)]
        self.last_layer = range(layer_start, layer_start + n_nodes)

    def compute(self, input_values):
        # set inputs
        for idx, value in enumerate(input_values):
            self.neurons[idx+1].output = value
        # propagate
        for neuron in self.neurons:
            neuron.propagate(self.neurons)
        # get output
        return neuron.output

    def train(self, data):
        for input_values, expect in data:
            # compute actual output
            output = self.compute(input_values)
            # initialiaze gradient
            self.neurons[-1].local_gradient = output - expect
            # backpropagate
            for neuron in reversed(self.neurons):
                neuron.backpropagate(self.neurons)


if __name__ == "__main__":
    nn = NeuralNetwork(2)
    nn.add_layer(20)
    nn.add_layer(1)

    f = (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0))
    nn.train(random.choice(f) for _ in range(10000))

    for input_values, expect in f:
        output = nn.compute(input_values)
        print(expect, output)

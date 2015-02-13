import random
import math


def sigmoid(x):
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

    def propagate(self, input_values):
        for idx, value in enumerate(input_values):
            self.neurons[idx+1].output = value
        for neuron in self.neurons:
            neuron.propagate(self.neurons)
        return neuron.output

    def backpropagate(self, correction):
        self.neurons[-1].local_gradient = correction
        for neuron in reversed(self.neurons):
            neuron.backpropagate(self.neurons)

    def train(self, data):
        for input_values, output in data:
            correction = self.propagate(input_values) - output
            self.backpropagate(correction)

    def compute(self, input_values):
        return self.propagate(input_values)


nn = NeuralNetwork(2)
nn.add_layer(10)
nn.add_layer(1)

f = (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0))
nn.train(random.choice(f) for _ in range(10000))

for input_values, output in f:
    print(output, nn.compute(input_values))

import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return (1 - s) * s


class Input:
    output = 1.
    grad = 0.
    inputs = []

    def propagate(self):
        pass

    def backpropagate(self):
        pass

neurons = [Input()]


class Neuron(Input):
    def __init__(self, inputs=[]):
        self.inputs = [(i, 1.) for i in [0] + inputs]
        self.output = 0.
        self.grad = 0.

    def propagate(self):
        self.output = sigmoid(sum(neurons[i].output*w for i, w in self.inputs))
        self.grad = 0.

    def backpropagate(self):
        self.inputs = [
            (i, w + self.grad*neurons[i].output) for i, w in self.inputs]
        grad = sigmoid_prime(self.grad)
        for i, _ in self.inputs:
            x = neurons[i]
            x.grad += x.output * grad

neurons.append(Input())
neurons.append(Input())
neurons.append(Neuron([1, 2]))

neuron = neurons[3]

f = (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1))
data = (random.choice(f) for _ in range(500))
for input_values, output in data:
    for idx, value in enumerate(input_values):
        neurons[idx+1].output = value
    neuron.propagate()
    neuron.grad = output - neuron.output
    neuron.backpropagate()

for input_values, output in f:
    for idx, value in enumerate(input_values):
        neurons[idx+1].output = value
    neuron.propagate()
    print(output, neuron.output)

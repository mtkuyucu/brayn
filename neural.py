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

    def propagate(self, neurons):
        pass

    def backpropagate(self, neurons):
        pass


class Neuron(Input):
    def __init__(self, inputs=[]):
        self.inputs = [(i, 1.) for i in [0] + inputs]
        self.output = 0.
        self.grad = 0.

    def propagate(self, neurons):
        self.output = sigmoid(sum(neurons[i].output*w for i, w in self.inputs))
        self.grad = 0.

    def backpropagate(self, neurons):
        self.inputs = [
            (i, w + self.grad*neurons[i].output) for i, w in self.inputs]
        grad = sigmoid_prime(self.grad)
        for i, _ in self.inputs:
            x = neurons[i]
            x.grad += x.output * grad


def propagate(neurons, input_values):
    for idx, value in enumerate(input_values):
        neurons[idx+1].output = value
    for neuron in neurons:
        neuron.propagate(neurons)
    return neuron.output


def backpropagate(neurons, correction):
    neurons[-1].grad = correction
    for neuron in reversed(neurons):
        neuron.backpropagate(neurons)


neurons = [Input()]  # normal input (1.)
neurons += [Input(), Input()]  # actual inputs
neurons += [Neuron([1, 2])]  # output

f = (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1))
data = (random.choice(f) for _ in range(500))
for input_values, output in data:
    correction = output - propagate(neurons, input_values)
    backpropagate(neurons, correction)

for input_values, output in f:
    print(output, propagate(neurons, input_values))

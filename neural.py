import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return (1 - s) * s


class DummyUnit:
    output = 1.
    grad = 0.
neurons = [DummyUnit()]


class Neuron:
    def __init__(self, inputs=[]):
        self.inputs = inputs + [0]
        self.weights = [1.] * len(self.inputs)
        self.output = 0.
        self.grad = 0.

    def propagate(self):
        self.output = sigmoid(sum(
            neurons[i].output*w for i, w in zip(self.inputs, self.weights)))
        self.grad = 0.

    def backpropagate(self):
        self.weights = [w + self.grad*neurons[i].output
                        for i, w in zip(self.inputs, self.weights)]
        grad = sigmoid_prime(self.grad)
        for i in self.inputs:
            x = neurons[i]
            x.grad += x.output * grad

neurons.append(Neuron())
neurons.append(Neuron())
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

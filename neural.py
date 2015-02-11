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
dummy_unit = DummyUnit()


class Neuron:
    def __init__(self, inputs=[]):
        self.inputs = inputs + [dummy_unit]
        self.weights = [1.] * len(self.inputs)
        self.output = 0.
        self.grad = 0.

    def propagate(self):
        self.output = sigmoid(sum(
            x.output*w for x, w in zip(self.inputs, self.weights)))
        self.grad = 0.

    def backpropagate(self):
        self.weights = [
            w + self.grad*x.output for x, w in zip(self.inputs, self.weights)]
        grad = sigmoid_prime(self.grad)
        for x in self.inputs:
            x.grad += x.output * grad

inputs = [Neuron(), Neuron()]
neuron = Neuron(inputs)

f = (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1))
data = (random.choice(f) for _ in range(500))
for input_values, output in data:
    inputs[0].output, inputs[1].output = input_values
    neuron.propagate()
    neuron.grad = output - neuron.output
    neuron.backpropagate()

for input_values, output in f:
    inputs[0].output, inputs[1].output = input_values
    neuron.propagate()
    print(output, neuron.output)

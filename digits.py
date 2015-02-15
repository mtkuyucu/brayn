import gzip
import struct

import neural


def labels_from(filename):
    with gzip.open(filename) as f:
        magicword, n_labels, = struct.unpack(">II", f.read(8))
        assert magicword == 2049
        for _ in range(n_labels):
            yield struct.unpack("B", f.read(1))[0]


def images_from(filename):
    with gzip.open(filename) as f:
        header = struct.unpack(">IIII", f.read(16))
        magicword, n_images, n_rows, n_cols = header
        assert magicword == 2051
        for _ in range(n_images):
            yield (byte / 256. for byte in f.read(n_rows * n_cols))


nn = neural.NeuralNetwork(28*28)
nn.add_layer(10)
nn.add_layer(1)

n_iterations = 1


def data():
    for _ in range(n_iterations):
        labels = labels_from("mnist/train-labels-idx1-ubyte.gz")
        images = images_from("mnist/train-images-idx3-ubyte.gz")
        for label, image in zip(labels, images):
            if label in (0, 5):
                yield image, float(label == 0)


def tests():
    labels = labels_from("mnist/t10k-labels-idx1-ubyte.gz")
    images = images_from("mnist/t10k-images-idx3-ubyte.gz")
    for label, image in zip(labels, images):
        if label in (0, 5):
            yield image, float(label == 0)

print("training")
nn.train(data())

print("testing")
n_tests, n_successes = 0, 0
for inputs_values, output in tests():
    n_tests += 1
    if (nn.compute(inputs_values) > 0.5) == (output > 0.5):
        n_successes += 1
print("%u / %u" % (n_successes, n_tests))

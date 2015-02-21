import neural
import mnist


nn = neural.NeuralNetwork(28*28)
nn.add_layer(10)
nn.add_layer(1)
n_iterations = 1


def data():
    for _ in range(n_iterations):
        labels = mnist.labels_from("mnist/train-labels-idx1-ubyte.gz")
        images = mnist.images_from("mnist/train-images-idx3-ubyte.gz")
        for label, image in zip(labels, images):
            image = (pixel / 256. for pixel in image)
            yield image, float(label == 0)


def tests():
    labels = mnist.labels_from("mnist/t10k-labels-idx1-ubyte.gz")
    images = mnist.images_from("mnist/t10k-images-idx3-ubyte.gz")
    for label, image in zip(labels, images):
        image = (pixel / 256. for pixel in image)
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

import neural
import mnist


nn = neural.NeuralNetwork(28*28)
nn.add_layer(10)
n_iterations = 1


def data():
    for _ in range(n_iterations):
        labels = mnist.labels_from("mnist/train-labels-idx1-ubyte.gz")
        images = mnist.images_from("mnist/train-images-idx3-ubyte.gz")
        for label, image in zip(labels, images):
            image = (pixel / 256. for pixel in image)
            expect = [0.]*label + [1.] + [0.]*(10-label)
            yield image, expect


def tests():
    labels = mnist.labels_from("mnist/t10k-labels-idx1-ubyte.gz")
    images = mnist.images_from("mnist/t10k-images-idx3-ubyte.gz")
    for label, image in zip(labels, images):
        image = (pixel / 256. for pixel in image)
        expect = [0.]*label + [1.] + [0.]*(10-label)
        yield image, expect


print("training")
for inputs, expect in data():
    nn.train(inputs, expect)

print("testing")
n_tests, n_successes = 0, 0
for inputs, expect in tests():
    n_tests += 1
    # run neural network
    output = nn.compute(inputs)
    # recover labels
    output = output.index(max(output))
    expect = expect.index(max(expect))
    # check result
    if output == expect:
        n_successes += 1
print("%u / %u" % (n_successes, n_tests))

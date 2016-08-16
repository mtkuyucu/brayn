import gzip
import struct
import os
mport file

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
            yield f.read(n_rows * n_cols)


def to_netpbm(filename, image):
    with open(filename, "wb") as f:
        f.write(b"P5\n28 28\n255\n" + image)


def export(path, labels, images):
    if not path:
        path = "."
    os.makedirs(path, exist_ok=True)
    for index, (label, image) in enumerate(zip(labels, images)):
        to_netpbm(path + "/%05u_%u.pgm" % (index, label), image)


if __name__ == "__main__":
    labels = labels_from("mnist/train-labels-idx1-ubyte.gz")
    images = images_from("mnist/train-images-idx3-ubyte.gz")
    export("mnist/train", labels, images)
    labels = labels_from("mnist/t10k-labels-idx1-ubyte.gz")
    images = images_from("mnist/t10k-images-idx3-ubyte.gz")
    export("mnist/test", labels, images)

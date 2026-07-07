"""MNIST loading, replicating bayesnn/mnist.py + nncfg.create_batch semantics."""

import gzip
import os
import struct
import urllib.request

import numpy as np

MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
FILES = {
    "training": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "testing": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}


def download(root):
    os.makedirs(root, exist_ok=True)
    for img, lbl in FILES.values():
        for name in (img, lbl):
            path = os.path.join(root, name)
            if os.path.exists(path):
                continue
            last_err = None
            for mirror in MIRRORS:
                url = mirror + name + ".gz"
                try:
                    print(f"downloading {url}")
                    with urllib.request.urlopen(url) as r:
                        data = gzip.decompress(r.read())
                    with open(path, "wb") as f:
                        f.write(data)
                    break
                except Exception as e:  # try next mirror
                    last_err = e
            else:
                raise RuntimeError(f"could not download {name}") from last_err


def load(dataset, path):
    """Load MNIST idx files (as bayesnn/mnist.py load(), all digits)."""
    fname_img, fname_lbl = FILES[dataset]
    with open(os.path.join(path, fname_lbl), "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.int8)
    with open(os.path.join(path, fname_img), "rb") as f:
        _, size, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
    return images, labels


def create_batch(images, labels, nbatch, rng, doshuffle=False, scale=1.0):
    """nncfg.create_batch: optional shuffle, drop remainder, batch, scale."""
    nsize = labels.shape[0] // nbatch * nbatch
    if labels.shape[0] % nbatch != 0:
        print(f"{labels.shape[0] % nbatch} data will be dropped during batching")
    if doshuffle:
        ind = rng.permutation(images.shape[0])
        images, labels = images[ind], labels[ind]
    images = images[:nsize]
    labels = labels[:nsize]
    xdata = np.float32(images.reshape(nsize // nbatch, nbatch, -1)) * scale
    ylabel = labels.reshape(nsize // nbatch, nbatch).astype(np.int32)
    return xdata, ylabel


def load_mnist_batched(path, batch_size, seed, nvalid=10000):
    """Full pipeline of bayesnn/mnist.py run_exp: batch, shuffle, split validation."""
    rng = np.random.RandomState(seed)
    train_images, train_labels = load("training", path)
    test_images, test_labels = load("testing", path)
    train_x, train_y = create_batch(train_images, train_labels, batch_size, rng,
                                    doshuffle=True, scale=1.0 / 256.0)
    test_x, test_y = create_batch(test_images, test_labels, batch_size, rng,
                                  doshuffle=True, scale=1.0 / 256.0)
    assert nvalid % batch_size == 0
    nv = nvalid // batch_size
    valid_x, valid_y = train_x[:nv], train_y[:nv]
    train_x, train_y = train_x[nv:], train_y[nv:]
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

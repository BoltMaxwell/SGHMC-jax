"""MovieLens ml-1m loading and fold splitting, replicating mf/python/{mkdata,mknfold}.py."""

import os
import random
import urllib.request
import zipfile

import numpy as np

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download(root):
    path = os.path.join(root, "ml-1m", "ratings.dat")
    if os.path.exists(path):
        return path
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "ml-1m.zip")
    if not os.path.exists(zip_path):
        print(f"downloading {ML1M_URL}")
        urllib.request.urlretrieve(ML1M_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract("ml-1m/ratings.dat", root)
    return path


def load_ratings(ratings_path):
    """mkdata.py: contiguous ids in first-seen order, labels (rating - 1) / 4."""
    umap, imap = {}, {}
    uids, iids, labels = [], [], []
    with open(ratings_path) as f:
        for line in f:
            arr = line.split("::")
            uid, iid, score = int(arr[0]), int(arr[1]), int(arr[2])
            uids.append(umap.setdefault(uid, len(umap)))
            iids.append(imap.setdefault(iid, len(imap)))
            labels.append((score - 1) / 4.0)
    uids = np.asarray(uids, np.int32)
    iids = np.asarray(iids, np.int32)
    labels = np.asarray(labels, np.float32)
    print(f"nuser={len(umap)}, nitem={len(imap)}, mean={labels.mean():.6f}")
    return uids, iids, labels, len(umap), len(imap)


def fold_assignment(n, nfold=5):
    """mknfold.py fold ids, reproducing Python 2 random.seed(0) + randint(1, nfold).

    Python 2's randint(a, b) is a + int(random() * (b - a + 1)); random() is the
    same MT19937 double generator in Python 3, so this reproduces the original
    fold membership bit-for-bit.
    """
    rng = random.Random(0)
    return np.fromiter((1 + int(rng.random() * nfold) for _ in range(n)),
                       np.int8, count=n)


def load_fold(ratings_path, fold, nfold=5):
    """Train/test split for one fold (fold's lines are test, rest train)."""
    uids, iids, labels, num_user, num_item = load_ratings(ratings_path)
    fa = fold_assignment(len(labels), nfold)
    test = fa == fold
    train = ~test
    data = {
        "train": (uids[train], iids[train], labels[train]),
        "test": (uids[test], iids[test], labels[test]),
        "num_user": num_user,
        "num_item": num_item,
        "base_score": float(labels.astype(np.float64).mean()),
    }
    print(f"fold {fold}: {train.sum()} train, {test.sum()} test")
    return data

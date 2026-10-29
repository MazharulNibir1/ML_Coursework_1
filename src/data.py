"""Data loading and fixed, reproducible partitions for Fashion-MNIST (Part 0).

All partitions are created once with SPLIT_SEED and saved to splits/splits.npz.
Later code only ever *loads* the saved indices, so the splits never change.
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_FILE = ROOT / "splits" / "splits.npz"

SPLIT_SEED = 708  # the only seed used for creating partitions

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_fashion_mnist():
    """Download (first time only) and return the official train and test sets.

    Images are returned as float32 in [0, 1] with shape (N, 784).
    Dividing by 255 is a fixed rescaling, not something learned from data,
    so it cannot leak information between splits.
    """
    from torchvision.datasets import FashionMNIST

    train = FashionMNIST(DATA_DIR, train=True, download=True)
    test = FashionMNIST(DATA_DIR, train=False, download=True)
    X_train = train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    X_test = test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    return X_train, y_train, X_test, y_test


def _stratified_sample(indices, labels, size, seed):
    """Pick `size` indices from `indices`, keeping the class proportions."""
    chosen, _ = train_test_split(indices, train_size=size, stratify=labels[indices],
                                 random_state=seed)
    return np.sort(chosen)


def make_splits(y_train_official, y_test_official, seed=SPLIT_SEED):
    """Create every partition used in the coursework.

    Labels are used only to stratify (keep class proportions); they never
    influence any model, distance or centroid.
    """
    all_idx = np.arange(len(y_train_official))
    train_idx, val_idx = train_test_split(all_idx, test_size=10_000,
                                          stratify=y_train_official, random_state=seed)
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)

    test_idx = np.arange(len(y_test_official))
    return {
        # indices into the OFFICIAL TRAIN set
        "train": train_idx,                                   # 50,000
        "val": val_idx,                                       # 10,000
        "train_sub": _stratified_sample(train_idx, y_train_official, 10_000, seed),
        "cluster_train": _stratified_sample(train_idx, y_train_official, 3_000, seed + 1),
        "cluster_val": _stratified_sample(val_idx, y_train_official, 1_000, seed + 2),
        # indices into the OFFICIAL TEST set
        "cluster_test": _stratified_sample(test_idx, y_test_official, 1_000, seed + 3),
        "seed": np.array([seed]),
    }


def get_splits(y_train_official, y_test_official):
    """Load the saved splits, creating them only if they do not exist yet."""
    if not SPLIT_FILE.exists():
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(SPLIT_FILE, **make_splits(y_train_official, y_test_official))
    saved = np.load(SPLIT_FILE)
    return {k: saved[k] for k in saved.files}

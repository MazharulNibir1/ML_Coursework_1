#!/usr/bin/env bash
# Download Fashion-MNIST (official files from the Zalando Research GitHub repo)
# into data/FashionMNIST/raw/. torchvision verifies the MD5 checksums on first load.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/FashionMNIST/raw
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  [ -f "data/FashionMNIST/raw/$f.gz" ] || curl -sSL -o "data/FashionMNIST/raw/$f.gz" \
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/$f.gz"
done
echo "Fashion-MNIST downloaded to data/FashionMNIST/raw/"

import os
import struct

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def read_idx_images(filename):
    with open(filename, 'rb') as f:
        # read the magic number and dimensions
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images


def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(data_dir):
    train_images = read_idx_images(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    train_labels = read_idx_labels(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    test_images = read_idx_images(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    test_labels = read_idx_labels(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    return train_images, train_labels, test_images, test_labels


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        # add channel dimension and normalize to [0, 1]
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


train_images, train_labels, test_images, test_labels = load_mnist('data')
print(train_images.shape, train_labels.shape)
train_dataset = MNISTDataset(train_images, train_labels)
test_dataset = MNISTDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000)

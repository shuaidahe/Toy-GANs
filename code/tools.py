
from torchvision import datasets, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(batch_size):
    """load MNIST dataset.
    Args:
        batch_size (int): size
    Returns:
        loader: train and test loader.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
    )
    dt_train = datasets.MNIST(
        root="../../data/", train=True, transform=transform, download=False
    )
    # print(dt_train.train_data.size(1),dt_train.train_data.size(2)) #(28,28)
    dt_test = datasets.MNIST(
        root="../../data/", train=False, transform=transform, download=False
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dt_train, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dt_test, batch_size=batch_size, shuffle=False
    )
    return dt_train,dt_test,train_loader, test_loader

def plot_some():
    """Show some figuires.
    """
    train_loader, _ = load_mnist(256)
    _ , (data_, label_) = next(enumerate(train_loader))
    plt.figure()
    for i in range(35):
        plt.subplot(5, 7, i + 1)
        plt.tight_layout()
        plt.imshow(data_[i][0], cmap="gray", interpolation="none")
        plt.title(f"label: {label_[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()
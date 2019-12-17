import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import time
from tools import load_mnist,plot_some

# Sample from a Uniform distribution
def input_sampler():
    """Generate noise samples.
    Returns:
        Tensor: A lambda funtion for sampling noise.
    """
    # data shape: (m,n)=(batch_size, input_size.)
    # np.random.uniform(0,1,size=[m,n])
    return lambda m, n: torch.tensor(
        np.random.randn(m, n), dtype=torch.float, requires_grad=True
    )


def extract(tensor_value):
    """Extract a list from tensor data.
    Args:
        tensor_value (tensor): input
    Returns:
        list: ?
    """
    return tensor_value.data.storage().tolist()


def shorter(v):
    """Round a float number.
    Args:
        v (float or a list of float): input
    Returns:
        just a input: shorter number.
    """
    if isinstance(v, list):
        return [round(i, 4) for i in v]
    return round(v, 4)


class Generator(nn.Module):
    """ Generator
    """
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.m1 = nn.Linear(input_size, 256)
        self.m2 = nn.Linear(self.m1.out_features, self.m1.out_features * 2)
        self.m3 = nn.Linear(self.m2.out_features, self.m2.out_features * 2)
        self.m4 = nn.Linear(self.m3.out_features, output_size)

    def forward(self, x):
        x = torch.relu(self.m1(x))
        x = torch.relu(self.m2(x))
        x = torch.relu(self.m3(x))
        x = torch.tanh(self.m4(x))  # !Becareful with the activation function. Here data range is [-1,1]
        return x


class Descriminator(nn.Module):
    """ Descriminator
    """
    def __init__(self, input_size, output_size):
        super(Descriminator, self).__init__()
        self.m1 = nn.Linear(input_size, 1024)
        self.m2 = nn.Linear(self.m1.out_features, self.m1.out_features // 2)
        self.m3 = nn.Linear(self.m2.out_features, self.m2.out_features // 2)
        self.m4 = nn.Linear(self.m3.out_features, output_size)

    def forward(self, x):
        x = torch.relu(self.m1(x))
        x = torch.relu(self.m2(x))
        x = torch.relu(self.m3(x))
        x = torch.sigmoid(self.m4(x))
        return x


def train_GAN():
    """Main training function.
    """
    # model_setting():
    batch_size = 128
    _,_,train_loader, test_loader = load_mnist(batch_size)

    z_dim = 77
    mnist_dim = 28 * 28

    gi_sampler = input_sampler()
    D = Descriminator(mnist_dim, 1)  # output the probability.
    G = Generator(z_dim, mnist_dim)
    print(D, "\n", G, "\n")

    n_epochs = 120

    criterion = nn.BCELoss()
    g_lr = 2e-4 # *learning rate should not be too large to void model collapsing.
    d_lr = 2e-4
    # sgd_momentum=0.8
    # d_optimizer=optim.SGD(D.parameters(),lr=d_lr,momentum=sgd_momentum)
    # g_optimizer=optim.SGD(G.parameters(),lr=g_lr,momentum=sgd_momentum)
    d_optimizer = optim.Adam(D.parameters(), lr=d_lr)
    g_optimizer = optim.Adam(G.parameters(), lr=g_lr)

    # training
    for epoch in range(n_epochs):
        for batch_idx, (data_, _) in enumerate(train_loader):  # * Do not need y here.
            # data_.max()=1,data_.min()=-1
            # Train D
            d_optimizer.zero_grad()

            # on real
            d_real_data = data_.view(-1, mnist_dim)
            # print(f'real data size: {d_real_data.shape}')
            d_real_label = torch.ones([d_real_data.shape[0], 1])
            # print(f'real data size: {d_real_label.shape}')
            d_real_decision = D(d_real_data)
            err_d_real = criterion(d_real_decision, d_real_label)

            # on fake
            d_fake_data = G(gi_sampler(batch_size, z_dim))  # !detach? input noise.
            # print(f'd fake data:{d_fake_data.shape}')
            d_fake_label = torch.zeros([d_fake_data.shape[0], 1])
            d_fake_decision = D(d_fake_data)
            err_d_fake = criterion(d_fake_decision, d_fake_label)
            err_d=err_d_real +err_d_fake
            err_d.backward()
            d_optimizer.step()  # Can also use: (err_real+err_fake).backward().

            # Train G on D's response
            g_optimizer.zero_grad()
            noise_g = gi_sampler(batch_size, z_dim)
            g_fake_data = G(noise_g)
            # *'Train G to pretend it's genuine' (equivalent to the fomula in Goodfellow,2014).
            g_fake_label = torch.ones([batch_size, 1])
            g_fake_decision = D(g_fake_data)
            err_g = criterion(g_fake_decision, g_fake_label)
            err_g.backward()
            g_optimizer.step()

            # if (epoch % 2 == 0) & (batch_idx % 50 == 0):
            if (batch_idx % 50 == 0):
                print( f"Epoch:{epoch}|Batch:{batch_idx}\
                    \tD (real_err:{shorter(extract(err_d_real)[0])},fake_err:{shorter(extract(err_d_fake)[0])})\
                    \tG (err:{shorter(extract(err_g)[0])})")
        # plot some samples
        # if PLOT & (epoch % 2 == 0):
        if PLOT:
            print("\tPlotting the real vs. generated data...")

            # fake samples
            sample_size = 16
            fake_samples = G(gi_sampler(sample_size, z_dim)).detach().numpy()
            fake_samples = fake_samples.reshape(sample_size, 28, 28)
            print(data_.shape, fake_samples.shape)# !torch.Size([96, 1, 28, 28]) (16, 28, 28) Why 96?

            plt.figure(figsize=(11.69, 8.27))  # the size of a A4 paper (in inches).
            for i in range(sample_size * 2):
                plt.subplot(4, sample_size // 2, i + 1)
                # plt.tight_layout()
                if i < sample_size:
                    # real samples
                    plt.imshow(data_[i][0], cmap="gray", interpolation="none")
                    plt.title(f"R-{i+1}", fontsize=12)
                    plt.xticks([])
                    plt.yticks([])
                else:
                    plt.imshow(
                        fake_samples[i - sample_size], cmap="gray", interpolation="none"
                    )
                    plt.title(f"F-{i-sample_size+1}", fontsize=12)
                    plt.xticks([])
                    plt.yticks([])
            time_now = time.strftime("%Y_%m_%d_%H:%M:%S")
            plt.savefig(
                dpi=300,
                format="png",
                fname=f"../results/GANs/mnist/real_vs_fake-{time_now}-Epoch-{epoch}.png",
            )
            # plt.show()
            plt.clf()
    if SAVE:
        torch.save(G.state_dict(), f'../models/GANs/G-{time_now}.pth')
        torch.save(D.state_dict(), f'../models/GANs/D-{time_now}.pth')


if __name__ == "__main__":
    PLOT = True  # set True to save demo.
    SAVE = True  # set True to save model.
    # plot_some()
    train_GAN()

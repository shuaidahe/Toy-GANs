import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import time
import matplotlib.pyplot as plt
from tools import load_mnist


class Autoencoder(nn.Module):
    def __init__(self, n_inp, n_encoded):
        super(Autoencoder, self).__init__()
        self.encoder_fc1 = nn.Linear(n_inp, 1024)
        self.encoder_fc2 = nn.Linear(1024, n_encoded)

        self.decoder_fc1 = nn.Linear(n_encoded, 1024)
        self.decoder_fc2 = nn.Linear(1024, n_inp)

    def forward(self, x):
        e = self.encode(x)
        d = self.decode(e)
        return e, d

    def encode(self, x):
        e = torch.sigmoid(self.encoder_fc1(x))
        e = torch.sigmoid(self.encoder_fc2(e))
        return e

    def decode(self, x):
        d = torch.sigmoid(self.decoder_fc1(x))
        d = torch.sigmoid(self.decoder_fc2(d))
        return d

class GMMN(nn.Module):
    def __init__(self, n_start, n_out):
        super(GMMN, self).__init__()
        self.fc1 = nn.Linear(n_start, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 784)
        self.fc5 = nn.Linear(784, n_out)

    def forward(self, samples):
        x = torch.relu(self.fc1(samples))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x


def train():
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloader
    BATCH_SIZE = 128
    dt_train,dt_test,train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)

    # define autoencoder network
    N_INP = 28*28
    ENCODED_SIZE = 32
    encoder_net = Autoencoder(N_INP, ENCODED_SIZE).to(device)
    encoder_optimizer = optim.Adam(encoder_net.parameters())
    # define the GMMN
    NOISE_SIZE = 20
    gmm_net = GMMN(NOISE_SIZE, ENCODED_SIZE).to(device)
    gmmn_optimizer = optim.Adam(gmm_net.parameters(), lr=0.001)

    AE_PATH='../models/GMMN/'
    GMMN_PATH='../models/GMMN/'

    def train_autoencoder():
        n_epoches_ae=50
        print(f'Training Autoencoder:')
        for ep in range(n_epoches_ae):
            avg_loss = 0
            for idx, (x, _) in enumerate(train_loader):
                x = x.view(x.size()[0], -1).to(device)
                _, decoded = encoder_net(x)
                loss = torch.sum((x - decoded) ** 2)
                encoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                avg_loss += loss.item()
            avg_loss /= (idx + 1)

            print(f'\tEpoch: {ep}, \taverage loss: {avg_loss}')
        print('\tAutoencoder has been trained!')

        if AE_PATH:
            torch.save(encoder_net.state_dict(), AE_PATH+f'encoder_net+{time.strftime("%Y_%m_%d_%H:%M:%S")}.pth')

    def get_scale_matrix(M, N):
        s1 = (torch.ones((N, 1)) * 1.0 / N).to(device)
        s2 = (torch.ones((M, 1)) * -1.0 / M).to(device)
        return torch.cat((s1, s2), 0)

    def train_one_step(x, samples, sigma=[1]):
        samples = samples.to(device)
        gen_samples = gmm_net(samples)
        X = torch.cat((gen_samples, x), 0)
        XX = torch.matmul(X, X.t())
        X2 = torch.sum(X * X, 1, keepdim=True)
        exp = XX - 0.5 * X2 - 0.5 * X2.t()

        M = gen_samples.size()[0]
        N = x.size()[0]
        s = get_scale_matrix(M, N)
        S = torch.matmul(s, s.t())

        loss = 0
        for v in sigma:
            kernel_val = torch.exp(exp / v)
            loss += torch.sum(S * kernel_val)

        loss = torch.sqrt(loss)

        gmmn_optimizer.zero_grad()
        loss.backward()
        gmmn_optimizer.step()

        return loss

    def train_gmmn():
        # training loop
        n_epoches_gmmn=100
        print(f'Training GMMN:')
        for ep in range(n_epoches_gmmn):
            avg_loss = 0
            for idx, (data_, _) in enumerate(train_loader):
                x_ = data_.view(data_.size()[0], -1)
                with torch.no_grad():
                    x_ = x_.to(device)
                    encoded_x = encoder_net.encode(x_)

                # uniform random noise between [-1, 1]
                random_noise = torch.rand((BATCH_SIZE, NOISE_SIZE)) * 2 - 1
                loss = train_one_step(encoded_x, random_noise)
                avg_loss += loss.item()

            avg_loss /= (idx + 1)
            print(f'\tEpoch: {ep}, \taverage loss: {avg_loss}')

            if PLOT:
                print("\t Plotting the real, reconstructed, and generated data...")
                sample_size = 16
                real_data=data_[:sample_size]

                # reconstructed image
                _, reconstructed_data= encoder_net(real_data.reshape(sample_size,-1))
                # plt.imshow(y.detach().squeeze().numpy().reshape(28, 28))
                reconstructed_data=reconstructed_data.detach().numpy().reshape(sample_size, 28, 28)

                # fake samples
                noise_ = torch.tensor(np.random.rand(sample_size, NOISE_SIZE),dtype=torch.float,requires_grad=True) * 2 - 1
                encoded_x = gmm_net(noise_)
                fake_samples = encoder_net.decode(encoded_x)
                fake_samples = fake_samples.detach().numpy().reshape(sample_size, 28, 28)
                # print(data_.shape, fake_samples.shape)# !torch.Size([96, 1, 28, 28]) (16, 28, 28) Why 96?

                # plt.figure(figsize=(11.69, 8.27))  # the size of a A4 paper (in inches).
                plt.figure(figsize=(12.5,9.5))  # the size of a A4 paper (in inches).
                for i in range(sample_size * 3):
                    plt.subplot(6, sample_size // 2, i + 1)
                    # plt.tight_layout()
                    if i < sample_size:
                        # real samples
                        plt.imshow(data_[i][0], cmap="gray", interpolation="none")
                        plt.title(f"O-{i+1}", fontsize=10,color='black')
                        plt.xticks([])
                        plt.yticks([])
                    elif sample_size <= i < 2*sample_size:
                        # reconstructed samples
                        plt.imshow(reconstructed_data[i - sample_size], cmap="gray", interpolation="none")
                        plt.title(f"R-{i-sample_size+1}", fontsize=10,color='blue')
                        plt.xticks([])
                        plt.yticks([])
                    else:
                        # fake samples
                        plt.imshow(fake_samples[i -2*sample_size], cmap="gray", interpolation="none")
                        plt.title(f"F-{i-sample_size+1}", fontsize=10,color='red')
                        plt.xticks([])
                        plt.yticks([])
                time_now = time.strftime("%Y_%m_%d_%H:%M:%S")
                plt.savefig(
                    dpi=300,
                    format="png",
                    fname=f"../results/GMMN/mnist/gmmn-real_vs_fake-{time_now}-Epoch-{ep}.png",
                )
                # plt.show()
                plt.clf()

        print('\tGMMN trained!')
        if GMMN_PATH:
            torch.save(gmm_net.state_dict(), GMMN_PATH+f'gmmn-{time.strftime("%Y_%m_%d_%H:%M:%S")}.pth')

    train_autoencoder()
    train_gmmn()

if __name__ == "__main__":
    PLOT=True
    train()
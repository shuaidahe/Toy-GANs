import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# real data
data_mean=5
data_std=1.5
def real_data_sampler(mu,sigma):
    return lambda n:torch.tensor(np.random.normal(mu,sigma,(1,n)),dtype=torch.float,requires_grad=False)

# Sample from a Uniform distribution
def input_sampler():
    # data shape: (m,n)=(batch_size, input_size.)
    return lambda m,n : torch.tensor(np.random.uniform(0,1,size=[m,n]),dtype=torch.float,requires_grad=False)

def extract(tensor_value):
    return tensor_value.data.storage().tolist()

def shorter(v):
    if isinstance(v,list):
        return [round(i,4) for i in v]
    else:
        return round(v,4)

def stats(d):
    return [np.mean(d),np.std(d)]

class Generator(nn.Module):
    def __init__(self,input_size,output_size):
        super(Generator,self).__init__()
        self.m1=nn.Linear(input_size,10)
        self.m2=nn.Linear(10,10)
        self.m3=nn.Linear(10,output_size)
    
    def forward(self, x):
        x=torch.relu(self.m1(x))
        x=torch.relu(self.m2(x))
        x=self.m3(x) # ! The output of GAN should not use an activation function.
        return x

class Descriminator(nn.Module):
    def __init__(self, input_size,output_size):
        super(Descriminator,self).__init__()
        self.m1=nn.Linear(input_size,10)
        self.m2=nn.Linear(10,10)
        self.m3=nn.Linear(10,output_size)
    
    def forward(self, x):
        x=torch.relu(self.m1(x))
        x=torch.relu(self.m2(x))
        x=torch.sigmoid(self.m3(x))
        return x

def train_GAN():
    # model_setting():
    batch_size=256
    g_input_size=1
    g_output_size=1
    d_input_size=batch_size
    d_output_size=1

    d_sampler=real_data_sampler(data_mean,data_std)
    gi_sampler=input_sampler()
    D=Descriminator(d_input_size,d_output_size)
    G=Generator(g_input_size,g_output_size)

    n_epochs=4000
    print_interval=100
    g_steps=20
    d_steps=20
    
    criterion=nn.BCELoss()
    g_lr=1e-3
    d_lr=1e-3
    # sgd_momentum=0.8
    # d_optimizer=optim.SGD(D.parameters(),lr=d_lr,momentum=sgd_momentum)
    # g_optimizer=optim.SGD(G.parameters(),lr=g_lr,momentum=sgd_momentum)
    d_optimizer=optim.Adam(D.parameters(),lr=d_lr)
    g_optimizer=optim.Adam(G.parameters(),lr=d_lr)

    # training
    for epoch in range(n_epochs):
        # Descriminator first
        for d_idx in range(d_steps):
            d_optimizer.zero_grad() # * Sometimes it's safer to use D.zero_grad()
            
            # Train D on real
            d_real_data=d_sampler(d_input_size)
            d_real_label=torch.ones([1,1])
            d_real_decision=D(d_real_data)
            err_d_real=criterion(d_real_decision,d_real_label)
            err_d_real.backward()

            # Train D on fake
            noise_d=gi_sampler(batch_size,g_input_size)
            d_fake_data=G(noise_d).detach() # !detach?
            # d_fake_data=G(noise_d) # ! detach?
            d_fake_label=torch.zeros([1,1])
            d_fake_decision=D(d_fake_data.t())
            err_d_fake=criterion(d_fake_decision,d_fake_label)
            err_d_fake.backward()
            d_optimizer.step() # Only optimizes D's parameters based on stored gradients from backward(). Can also use: (err_real+err_fake).backward().
        
        for g_idx in range(g_steps):
            g_optimizer.zero_grad() 
            # Train G on D's response
            noise_g=gi_sampler(batch_size,g_input_size)
            g_fake_data=G(noise_g)
            g_fake_label=torch.ones([1,1]) # !train G to pretend it's genuine (equivalent to the fomula in Goodfellow,2014).
            g_fake_decision=D(g_fake_data.t())
            err_g=criterion(g_fake_decision,g_fake_label)
            err_g.backward()
            g_optimizer.step()

        if epoch % print_interval==0:
            print(f'Epoch:{epoch}')
            print(f'\t D(real_err:{shorter(extract(err_d_real)[0])} fake_err:{shorter(extract(err_d_fake)[0])}; G(err:{shorter(extract(err_g)[0])})')
            print(f'\t Real distribution({shorter(stats(extract(d_real_data)))}); Fake distribution({shorter(stats(extract(d_fake_data)))}).')

    if plot:
        print("The real vs. generated:")

        sample_size=5000
        real_samples=d_sampler(sample_size)
        y_real = shorter(extract(real_samples))
        print(f'Real samples {y_real}')

        noise_samples=gi_sampler(sample_size,1)
        fake_samples=G(noise_samples)
        y_fake = shorter(extract(fake_samples))
        print(f'Generated samples {y_fake}')

        fig,ax=plt.subplots(2,1,figsize=(6,6))
        plt.subplots_adjust(wspace=None, hspace=0.3)
        # ax[0].text(1,6, f'Sample size = {sample_size}'j, style='italic')
        ax[0].hist(y_real,bins=80)
        ax[0].set_xlabel('y_real')
        ax[0].set_ylabel('Count)')
        ax[0].set_title(f'Histogram of Real Samples (size = {sample_size})')
  
        ax[1].hist(y_fake,bins=80)
        ax[1].set_xlabel('y_fake')
        ax[1].set_ylabel('Count')
        ax[1].set_title(f'Histogram of Generated Samples (size = {sample_size})')

        time_now=time.strftime('%Y_%m_%d_%H:%M:%S')
        plt.savefig(dpi=300,format='pdf',fname=f'./results/base_model-Generated_samples-{time_now}.pdf')
        plt.show()

if __name__ == "__main__":
    plot=True
    train_GAN()
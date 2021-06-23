#%%
import torch.nn as nn
import torch


class network:
    def __init__(self):
        self.nc_im = 3
        self.ker_size = 3
        self.stride = 1
        self.padd_size = 1
        self.minKerNum = 32
        self.startKerNum = 256
        self.num_layer = 5


network = network()


def ConvBlock(in_channel, out_channel, ker_size, stride, padding):
    block = nn.Sequential()
    block.add_module(
        "conv",
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=ker_size,
            stride=stride,
            padding=padding,
        ),
    ),
    block.add_module("norm", nn.BatchNorm2d(out_channel)),
    block.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
    return block


def ConvBlock_i(in_channel, out_channel, ker_size, stride, padding):
    block = nn.Sequential()
    block.add_module(
        "conv",
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=ker_size,
            stride=stride,
            padding=padding,
        ),
    ),
    block.add_module("norm", nn.InstanceNorm2d(out_channel)),
    block.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
    return block


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        N = network.startKerNum
        self.head = ConvBlock_i(
            network.nc_im, N, network.ker_size, network.stride, network.padd_size,
        )
        self.body = nn.Sequential()
        for i in range(network.num_layer - 2):
            N = int(N / 2)
            block = ConvBlock_i(
                max(2 * N, network.minKerNum),
                max(N, network.minKerNum),
                network.ker_size,
                network.stride,
                network.padd_size,
            )

            self.body.add_module("block{}".format(i), block)
        self.tail = nn.Sequential(
            nn.Conv2d(
                max(N, network.minKerNum),
                network.nc_im,
                kernel_size=network.ker_size,
                stride=network.stride,
                padding=network.padd_size,
            ),
            nn.Tanh(),
        )

    def forward(self, noise, img):
        x = img + noise
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return torch.clamp(x + img, -1, 1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        N = network.startKerNum
        self.head = ConvBlock(
            network.nc_im, N, network.ker_size, network.stride, network.padd_size,
        )
        self.body = nn.Sequential()
        for i in range(network.num_layer - 2):
            N = int(N / 2)
            block = ConvBlock(
                max(2 * N, network.minKerNum),
                max(N, network.minKerNum),
                network.ker_size,
                network.stride,
                network.padd_size,
            )

            self.body.add_module("block{}".format(i), block)

        self.tail = nn.Conv2d(
            max(N, network.minKerNum),
            1,
            kernel_size=network.ker_size,
            stride=network.stride,
            padding=network.padd_size,
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


# from config import *

# a = Generator().to(device)
# print(a)
# b = Discriminator().to(device)
# print(b)
# #%%

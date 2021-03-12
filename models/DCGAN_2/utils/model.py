import torch
import torch.nn as nn

from .constants import *

# Given transposed=1, weight of size [100, 512, 4, 4], expected input[8, 3, 256, 256] to have 100 channels, but got 3 channels instead

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 256

# Size of feature maps in discriminator
ndf = 256


class Generator(nn.Module):
    def __init__(self, filter=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(3, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Conv2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Conv2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.Conv2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Conv2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# class Generator(nn.Module):
#     def __init__(self, filter=64):
#         super(Generator, self).__init__()
#         self.downsamples = nn.ModuleList([
#             # (b, filter, 128, 128)
#             Downsample(3, filter, kernel_size=4, apply_instancenorm=False),
#             Downsample(filter, filter * 2),  # (b, filter * 2, 64, 64)
#             Downsample(filter * 2, filter * 4),  # (b, filter * 4, 32, 32)
#             Downsample(filter * 4, filter * 8),  # (b, filter * 8, 16, 16)
#             Downsample(filter * 8, filter * 8),  # (b, filter * 8, 8, 8)
#         ])

#         self.upsamples = nn.ModuleList([
#             Upsample(filter * 8, filter * 8),
#             Upsample(filter * 16, filter * 4, dropout=False),
#             Upsample(filter * 8, filter * 2, dropout=False),
#             Upsample(filter * 4, filter, dropout=False)
#         ])

#         self.last = nn.Sequential(
#             nn.ConvTranspose2d(filter * 2, 3, kernel_size=4,
#                                stride=2, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         skips = []
#         for l in self.downsamples:
#             x = l(x)
#             skips.append(x)

#         skips = reversed(skips[:-1])
#         for l, s in zip(self.upsamples, skips):
#             x = l(x, s)

#         out = self.last(x)

#         return out


class Discriminator(nn.Module):
    def __init__(self, filter=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# class Discriminator(nn.Module):
#     def __init__(self, filter=64):
#         super(Discriminator, self).__init__()

#         self.block = nn.Sequential(
#             Downsample(3, filter, kernel_size=4, stride=2,
#                        apply_instancenorm=False),
#             Downsample(filter, filter * 2, kernel_size=4, stride=2),
#             Downsample(filter * 2, filter * 4, kernel_size=4, stride=2),
#             Downsample(filter * 4, filter * 8, kernel_size=4, stride=1),
#         )

#         self.last = nn.Conv2d(
#             filter * 8, 1, kernel_size=4, stride=1, padding=1)

#     def forward(self, x):
#         x = self.block(x)
#         x = self.last(x)

#         return x

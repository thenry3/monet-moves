import torch
import torch.nn as nn
import torch.nn.functional as F

# latent_size = 256 * 256

# generator = nn.Sequential(
#     # in: latent_size x 1 x 1

#     nn.Conv2d(3, 512, kernel_size=4,
#                        stride=1, padding=0, bias=False),
#     nn.BatchNorm2d(512),
#     nn.ReLU(True),
#     # out: 512 x 4 x 4

#     nn.Conv2d(512, 256, kernel_size=4,
#                        stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(256),
#     nn.ReLU(True),
#     # out: 256 x 8 x 8

#     nn.Conv2d(256, 128, kernel_size=4,
#                        stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(128),
#     nn.ReLU(True),
#     # out: 128 x 16 x 16

#     nn.Conv2d(128, 64, kernel_size=4,
#                        stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(64),
#     nn.ReLU(True),
#     # out: 64 x 32 x 32

#     nn.Conv2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.Tanh()
#     # out: 3 x 64 x 64
# )

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(3, 512, kernel_size=4,
                       stride=1, padding=0, bias=False)
        self.bnorm1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=4,
                       stride=2, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 128, kernel_size=4,
                       stride=2, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=4,
                       stride=2, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.bnorm1(self.conv1(x))
        x = F.relu(x)
        x = self.bnorm2(self.conv2(x))
        x = F.relu(x)
        x = self.bnorm3(self.conv3(x))
        x = F.relu(x)
        x = self.bnorm4(self.conv4(x))
        x = F.relu(x)
        return self.tanh(self.conv5(x))


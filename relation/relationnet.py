import torch.nn as nn
from utils.conv_block import ConvBlock


class RelationNet(nn.Module):
    """
    Model as described in the reference paper,
    """

    def __init__(self, input_dim, hid_dim=64, z_dim=64):
        super().__init__()
        self.block1 = ConvBlock(input_dim, hid_dim, 3)
        self.block2 = ConvBlock(hid_dim, hid_dim, 3)
        self.block3 = ConvBlock(hid_dim, hid_dim, 3, max_pool=None)
        self.block4 = ConvBlock(hid_dim, z_dim, 3, max_pool=None)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.size(0), -1)

        return out

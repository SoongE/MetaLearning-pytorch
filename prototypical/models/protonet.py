import torch.nn as nn
from utils.conv_block import ConvBlock


class ProtoNet(nn.Module):
    """
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """

    def __init__(self, input_dim, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.block1 = ConvBlock(input_dim, hid_dim, 3, padding=1)
        self.block2 = ConvBlock(hid_dim, hid_dim, 3, padding=1)
        self.block3 = ConvBlock(hid_dim, hid_dim, 3, padding=1)
        self.block4 = ConvBlock(hid_dim, z_dim, 3, padding=1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.size(0), -1)

        return out

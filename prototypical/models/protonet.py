import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ProtoNet(nn.Module):
    """
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """

    def __init__(self, input_dim, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.block1 = conv_block(input_dim, hid_dim)
        self.block2 = conv_block(hid_dim, hid_dim)
        self.block3 = conv_block(hid_dim, hid_dim)
        self.block4 = conv_block(hid_dim, z_dim)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.size(0), -1)

        return out


if __name__ == "__main__":
    model = ProtoNet(3)

    print(model)
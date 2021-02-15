import torch.nn as nn
from utils.conv_block import ConvBlock


class Embedding(nn.Module):
    """
    Model as described in the reference paper,
    """

    def __init__(self, in_channel=3, z_dim=64):
        super().__init__()
        self.block1 = ConvBlock(in_channel, z_dim, 3, max_pool=2)
        self.block2 = ConvBlock(z_dim, z_dim, 3, max_pool=2)
        self.block3 = ConvBlock(z_dim, z_dim, 3, max_pool=None, padding=1)
        self.block4 = ConvBlock(z_dim, z_dim, 3, max_pool=None, padding=1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        return out


class RelationNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        if feature_dim == 64:
            self.layer1 = ConvBlock(128, 64, 3, max_pool=2, padding=1)
            self.layer2 = ConvBlock(64, 64, 3, max_pool=2, padding=1)
        else:
            self.layer1 = ConvBlock(128, 64, 3, max_pool=2)
            self.layer2 = ConvBlock(64, 64, 3, max_pool=2)

        self.fc1 = nn.Linear(feature_dim, 8)
        self.fc2 = nn.Linear(8, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))

        return out


if __name__ == '__main__':
    import torch

    class_per_it = 5
    num_support = 5
    num_query = 15

    embedding = Embedding(in_channel=3)
    model = RelationNetwork(64 * 3 * 3)

    out1 = embedding(torch.rand((class_per_it * num_support, 3, 84, 84)))
    out2 = embedding(torch.rand((class_per_it * num_query, 3, 84, 84)))

    out1_size = out1.size()

    out1 = out1.view(class_per_it, num_support, out1_size[1], out1_size[2], out1_size[3]).sum(dim=1)

    out1 = out1.repeat(class_per_it * num_query, 1, 1, 1)
    out2 = out2.repeat(class_per_it, 1, 1, 1)

    concat = torch.cat((out1, out2), dim=1)

    final = model(concat).view(-1, class_per_it)

    import random

    label = []
    for _ in range(75):
        label.append(random.randint(0, 3))

    label = torch.tensor(label).unsqueeze(1)

    # print(label.shape)
    # print(num_query*class_per_it,class_per_it)

    one_hot = torch.zeros(num_query * class_per_it, class_per_it).scatter_(1, label, 1)

    criterion = nn.MSELoss()
    loss = criterion(final, one_hot)
    # print(loss)

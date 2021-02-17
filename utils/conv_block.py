import torch


class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True,
                 use_bn=True,
                 max_pool=None,
                 activation="relu"):
        super().__init__()
        self.use_bn = use_bn
        self.use_max_pool = (max_pool is not None)
        self.use_activation = (activation is not None)

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        if self.use_activation:
            self.activation = get_activation(activation)
        if self.use_max_pool:
            self.max_pool = torch.nn.MaxPool2d(max_pool)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        if self.use_max_pool:
            x = self.max_pool(x)
        return x


def get_activation(activation):
    activation = activation.lower()
    if activation == "relu":
        return torch.nn.ReLU(inplace=True)
    elif activation == "relu6":
        return torch.nn.ReLU6(inplace=True)
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()

    else:
        raise NotImplementedError("Activation {} not implemented".format(activation))

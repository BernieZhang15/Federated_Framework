import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class logistic(nn.Module):
    def __init__(self, in_size=32 * 32 * 1, num_classes=10):
        super(logistic, self).__init__()
        self.linear = nn.Linear(in_size, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear(out)
        return out


class lstm(nn.Module):
    def __init__(self, input_size=32, hidden_size=128, num_layers=2, num_classes=10):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1, self.input_size, self.input_size)
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VGG(nn.Module):
    """VGG model """

    def __init__(self, features, size=512, out=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Linear(size, out),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11s():
    return VGG(make_layers([32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M']), size=128)


def vgg11():
    return VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']))


def conv(in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
    c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
    if batch_norm:
        bn = nn.GroupNorm(16, out_channels)
        # bn = nn.BatchNorm2d(out_channels)
        return nn.Sequential(c, bn)
    return c


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv_in = conv(self.in_channels, self.out_channels)
        self.conv_out = conv(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_out(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2)

        self.conv_in = conv(2 * self.out_channels, self.out_channels)
        self.conv_out = conv(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up, output_size=from_down.size())
        x = torch.cat((from_up, from_down), 1)
        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_out(x))
        return x


class SegUNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, depth=5, start_filts=64):
        super(SegUNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []

        outs = 0
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.conv_final = conv(outs, self.num_classes, kernel_size=1, padding=0, batch_norm=False)

    def forward(self, rgb):
        encoder_outs = []
        for i, module_down in enumerate(self.down_convs):
            rgb, before_pool_rgb = module_down(rgb)
            encoder_outs.append(before_pool_rgb)
        fm = rgb
        for i, module_up in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            rgb = module_up(before_pool, rgb)
        x = self.conv_final(rgb)
        return x, fm


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        group_num = 16
        if output_dim == 2:
            group_num = 1

        self.conv_block = nn.Sequential(
            # nn.BatchNorm2d(input_dim),
            nn.GroupNorm(group_num, input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            # nn.BatchNorm2d(output_dim),
            nn.GroupNorm(group_num, output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            # nn.BatchNorm2d(output_dim),
            nn.GroupNorm(group_num, output_dim)
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class ResUNet(nn.Module):
    def __init__(self, channel=3, filters=None):
        super(ResUNet, self).__init__()

        if filters is None:
            filters = [32, 64, 128, 256, 512]
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            # nn.BatchNorm2d(filters[0]),
            nn.GroupNorm(16, filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)
        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.up_residual_conv5 = ResidualConv(filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            ResidualConv(filters[0], 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, m):
        x1 = self.input_layer(m) + self.input_skip(m)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        x5 = self.bridge(x4)

        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)
        x7 = self.up_residual_conv1(x6)
        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)
        x9 = self.up_residual_conv2(x8)
        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)
        x11 = self.up_residual_conv3(x10)
        x11 = self.upsample_4(x11)
        x12 = torch.cat([x11, x1], dim=1)
        x13 = self.up_residual_conv4(x12)
        x14 = self.up_residual_conv5(x13)
        output = self.output_layer(x14)

        return output

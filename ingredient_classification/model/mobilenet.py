import torch.nn as nn

class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(Conv2dNormActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        if expand_ratio != 1:
            layers.append(Conv2dNormActivation(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0))
        layers.extend([
            Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out += x
        return out


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, width_multiplier=1.0, inverted_residual_setting=None):
        super(MobileNet, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        input_channel = int(input_channel * width_multiplier)
        self.last_channel = int(last_channel * width_multiplier) if width_multiplier > 1.0 else last_channel
        features = [Conv2dNormActivation(3, input_channel, kernel_size=3, stride=2, padding=1)]
        
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        # building last several layers
        features.append(Conv2dNormActivation(input_channel, self.last_channel, kernel_size=1, stride=1, padding=0))
        
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # global pooling
        x = self.classifier(x)
        return x
    
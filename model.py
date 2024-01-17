from torch.nn import Sequential, Conv2d, MaxPool2d, SELU, Linear, Flatten, Module, ReLU


class VGG16(Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.conv2 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.conv3 = Sequential(
            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.conv4 = Sequential(
            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.conv5 = Sequential(
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SELU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # AdaptiveAvgPool2d(output_size=(7, 7)),
        )
        self.dense = Sequential(
            Flatten(),
            Linear(in_features=4608, out_features=1024, bias=True),
            SELU(inplace=True),
            # Dropout(p=0.5, inplace=False),
            Linear(in_features=1024, out_features=512, bias=True),
            SELU(inplace=True),
            # Dropout(p=0.5, inplace=False),
            Linear(in_features=512, out_features=3, bias=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print("************x.shape:************", x.shape)
        x = self.dense(x)
        return x

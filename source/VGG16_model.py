import torch.nn as nn
from torchsummary import summary

VGG16_feature_extract = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512,  512, 'M']

class VGG16_net(nn.Module):
    def __init__(self, image_size=224, in_channels=3, numclasses=100):
        super(VGG16_net, self).__init__()
        self.in_channels = in_channels
        self.feature_extract_net = self.create_feature_extract(VGG16_feature_extract)

        out_feature_extract_size = int(512*image_size*image_size/1024)
        self.classification_net = nn.Sequential(
            nn.Linear(out_feature_extract_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, numclasses)
        )

    def forward(self,x):
        x = self.feature_extract_net(x)
        x = x.reshape(x.shape[0],-1)
        x = self.classification_net(x)
        return x

    def conv_layer(self, in_channels, out_channels):
        return [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]

    def create_feature_extract(self, architecture):
        layers = []
        # get input channels of image
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                # output channels is number of channels in architecture list element
                out_channels = x
                layers += self.conv_layer(in_channels, out_channels)
                # after convolution, new in_channels is current out_channels
                in_channels = x

            # if x is max pooling layer
            else:
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        return nn.Sequential(*layers)

if __name__ == '__main__':
    model = VGG16_net(in_channels=3, numclasses=100)
    summary(model, (3,224,224))



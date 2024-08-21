import torch
import torch.nn as nn
import torch.nn.functional as F
# from options import HiDDenConfiguration
# from model.conv_bn_relu import ConvBNRelu


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, decoder_channels=64, decoder_blocks=6, message_length=8):

        super(Decoder, self).__init__()
        self.channels = decoder_channels

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(message_length, message_length)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x

class Decoder_sigmoid(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, in_channels=3, decoder_channels=64, decoder_blocks=6, message_length=8):

        super(Decoder_sigmoid, self).__init__()
        self.channels = decoder_channels

        layers = [ConvBNRelu(in_channels, self.channels)]
        for _ in range(decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(message_length, message_length)
        self.out = nn.Sigmoid()

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        x = self.out(x)
        return x


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)



class Decoder_LN(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, decoder_channels=64, decoder_blocks=6, message_length=8):

        super(Decoder_LN, self).__init__()
        self.channels = decoder_channels

        layers = [ConvBNRelu_LN(3, self.channels)]
        for _ in range(decoder_blocks - 1):
            layers.append(ConvBNRelu_LN(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu_LN(self.channels, message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(message_length, message_length)
        self.out = nn.Sigmoid()

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        x = self.out(x)
        return x

class ConvBNRelu_LN(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu_LN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.LayerNorm((channels_out, 64, 64)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Decoder_sigmoid_classifier(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, in_channels=4, decoder_channels=64, decoder_blocks=6, message_length=8):

        super().__init__()
        self.channels = decoder_channels

        layers = [ConvBNRelu(in_channels, self.channels)]
        for _ in range(decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(message_length, message_length)
        self.out = nn.Sigmoid()

    def forward(self, image_with_wm, c):

        c = c.unsqueeze(-1).unsqueeze(-1)
        c = F.interpolate(c, size=(image_with_wm.shape[2], image_with_wm.shape[3]))
        image_with_wm = torch.cat( (image_with_wm,c), 1)

        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)

        # c = F.interpolate(c,size=(wm_resize, wm_resize))

        x = self.linear(x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    import torch
    c = torch.randn(1).unsqueeze(0)
    x = torch.randn(1,3,256,256)
    decoder = Decoder_sigmoid(decoder_channels=64, decoder_blocks=6, message_length=8)
    decoder = Decoder_sigmoid_classifier(decoder_channels=64, decoder_blocks=6, message_length=8)
    out = decoder(x, c)
    print()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

""" Originally developed by XD_XD in the SpaceNet 4 Challenge , then integrated
by Solaris project. We modified the code to accomodate different number of
input channels, e.g. 5-channel RGB+LIDAR input images.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg16


def get_modified_vgg16_unet(in_channels=3):
    """ Get a modified VGG16-Unet model with customized input channel numbers.
    For example, we can set in_channels=3 and input RGB 3-channel images.
    On the other hand, we can set in_channels=5 if we want to input both RGB
    and 2-channel LIDAR data (elevation + intensity).
    """
    class Modified_VGG16Unet(VGG16Unet):
        def __init__(self):
            super().__init__(in_channels=in_channels)
    return Modified_VGG16Unet


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class VGG16Unet(nn.Module):
    def __init__(self, in_channels=3, num_filters=32, pretrained=False):
        super().__init__()
        # Get VGG16 net as encoder
        self.encoder = vgg16(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        # Modify encoder architecture
        self.encoder[0] = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            self.encoder[0], self.relu, self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(
            self.encoder[10], self.relu, self.encoder[12], self.relu,
            self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(
            self.encoder[17], self.relu, self.encoder[19], self.relu,
            self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(
            self.encoder[24], self.relu, self.encoder[26], self.relu,
            self.encoder[28], self.relu)

        # Build decoder
        self.center = DecoderBlock(
            512, num_filters*8*2, num_filters*8)
        self.dec5 = DecoderBlock(
            512 + num_filters*8, num_filters*8*2, num_filters*8)
        self.dec4 = DecoderBlock(
            512 + num_filters*8, num_filters*8*2, num_filters*8)
        self.dec3 = DecoderBlock(
            256 + num_filters*8, num_filters*4*2, num_filters*2)
        self.dec2 = DecoderBlock(
            128 + num_filters*2, num_filters*2*2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        # Final output layer outputs logits, not probability
        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)
        
        # Final output layer outputs logits, not probability
        self.mask = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x, use_mask=False):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        x_out = self.final(dec1)
        if use_mask:
            mask_out = F.sigmoid(self.mask(dec1))
            return x_out, mask_out
        return x_out
    
class VGG16UNetWM(nn.Module):
    def __init__(self, in_channel=1, out_channel=3, num_filters=32, pretrained=True, requires_grad=True,args=None):

        super().__init__()
        self.out_channel = out_channel

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = vgg16(pretrained=pretrained).features#

        self.relu = nn.ReLU(inplace=True)
    
        self.encoder_conv1_1=nn.Conv2d(in_channel, 64, 3, padding=(1, 1))#
        self.conv1 = nn.Sequential(self.encoder_conv1_1,
                                self.relu,
                                self.encoder[2],
                                self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        # self.conv5 = nn.Sequential(self.encoder[24],
        #                            self.relu,
        #                            self.encoder[26],
        #                            self.relu,
        #                            self.encoder[28],
        #                            self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        # self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters + 1, out_channel, kernel_size=1)

        self.c_pool = nn.AdaptiveMaxPool2d((2,2))
        self.classifer_fc = nn.Sequential(nn.Linear(512*2*2,(512*2*2)//8),
                                          self.relu,
                                          nn.Linear((512*2*2)//8,1)
                                          )

    def forward(self, x, c=None, wm_resize=128):

        x = F.interpolate(x,size=(wm_resize, wm_resize))
        
        if c is not None:
            c = c.unsqueeze(-1).unsqueeze(-1)
            c = F.interpolate(c,size=(wm_resize, wm_resize))
            x = torch.cat( (x,c), 1)

        conv1 = self.conv1(x) # torch.Size([2, 64, 756, 1008])
        conv2 = self.conv2(self.pool(conv1)) # torch.Size([2, 128, 378, 504])
        conv3 = self.conv3(self.pool(conv2)) # torch.Size([2, 256, 189, 252])
        conv4 = self.conv4(self.pool(conv3)) # torch.Size([2, 512, 94, 126])
        # conv5 = self.conv5(self.pool(conv4))
        # center = self.center(self.pool(conv5))\
        center = self.center(self.pool(conv4)) # torch.Size([2, 256, 94, 126]) center <> upsampling module 
        x_c = self.c_pool(conv4) #2,512,2,2
        x_c = F.sigmoid(self.classifer_fc(x_c.view(x_c.shape[0],-1)))

        # dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([center, conv4], 1)) # torch.Size([2, 256, 188, 252])
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        c = x_c.unsqueeze(-1).unsqueeze(-1)
        c = F.interpolate(c,size=(wm_resize, wm_resize))
        dec1 = torch.cat((dec1, c), 1)
        x_out=self.final(dec1)
        
        return x_out, x_c

class WMDecoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=3, num_filters=32, decoder_fc_out=8, pretrained=True):

        super().__init__()
        self.out_channel = out_channel

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
    
        self.encoder_conv1_1=nn.Conv2d(in_channel, 64, 3, padding=(1, 1))#
        self.conv1 = nn.Sequential(self.encoder_conv1_1,
                                self.relu,
                                self.encoder[2],
                                self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.c_pool = nn.AdaptiveMaxPool2d((2,2))
        self.fc = nn.Sequential(nn.Linear(512*2*2,(512*2*2)//8),
                                          self.relu,
                                          nn.Linear((512*2*2)//8, 256),
                                          self.relu
                                          )
        self.classifer_fc = nn.Linear(256, 1)
        self.decoder_fc_1 = nn.Linear(256, 128)
        self.decoder_fc = nn.Linear(128, decoder_fc_out)

    def forward(self, x, wm_resize=None):

        if wm_resize:
            x = F.interpolate(x, size=(wm_resize, wm_resize))
        
        conv1 = self.conv1(x) # torch.Size([2, 64, 756, 1008])
        conv2 = self.conv2(self.pool(conv1)) # torch.Size([2, 128, 378, 504])
        conv3 = self.conv3(self.pool(conv2)) # torch.Size([2, 256, 189, 252])
        conv4 = self.conv4(self.pool(conv3)) # torch.Size([2, 512, 94, 126])
        conv5 = self.conv5(self.pool(conv4))
        x_cpool = self.c_pool(conv5) #2,512,2,2
        
        x_fc = self.fc(x_cpool.view(x_cpool.shape[0], -1))
        x_c = F.sigmoid(self.classifer_fc(x_fc))
        # tmp = torch.cat((x_fc, x_c), -1)
        tmp = torch.mul(x_fc, x_c)
        x_dec = F.sigmoid(self.decoder_fc(self.decoder_fc_1(tmp)))
        
        # mean conv
        # x_c = torch.mean(conv5.view(conv5.size(0), -1), dim=1)

        return x_dec, x_c


if __name__ == "__main__":
    unet_wm = VGG16UNetWM()
    x = torch.randn(1, 1, 256, 256)
    x_out, x_c = unet_wm(x, wm_resize=64)
    print()

    wm_decoder = WMDecoder()
    x = torch.randn(1, 1, 256, 256)
    x_dec, x_c = wm_decoder(x, wm_resize=64)
    print()

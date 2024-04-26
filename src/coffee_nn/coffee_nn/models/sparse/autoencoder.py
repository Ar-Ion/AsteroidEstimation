import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

class AutoEncoder(ME.MinkowskiNetwork):

    def __init__(self, D=2):
        super(AutoEncoder, self).__init__(D)

        self._D = D

        self.conv1 = self.create_downsampling_conv(1, 64)
        self.conv2 = self.create_downsampling_conv(64, 128)
        self.conv3 = self.create_downsampling_conv(128, 256)
        self.conv4 = self.create_upsampling_conv(256, 256)
        self.conv5 = self.create_upsampling_conv(256, 256)
        self.conv6 = self.create_upsampling_conv(256, 256)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
                
        return out
    
    def create_mlp(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiLinear(nin, nout, bias=False),
            ME.MinkowskiBatchNorm(nout),
            ME.MinkowskiReLU()
        )
    
    def create_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=nin,
                out_channels=nout,
                kernel_size=3,
                dilation=1,
                stride=1,
                bias=False,
                dimension=self._D),
            ME.MinkowskiBatchNorm(nout),
            ME.MinkowskiReLU()
        )
    
    def create_downsampling_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=nin,
                out_channels=nout,
                kernel_size=3,
                dilation=1,
                stride=1,
                bias=False,
                dimension=self._D),
            ME.MinkowskiBatchNorm(nout),
            ME.MinkowskiReLU(),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=self._D),
        )
    
    def create_upsampling_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=nin,
                out_channels=nout,
                kernel_size=3,
                dilation=1,
                stride=1,
                bias=False,
                dimension=self._D),
            ME.MinkowskiBatchNorm(nout),
            ME.MinkowskiReLU(),
            ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=self._D),
        )
    
    def create_mapping_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=nin,
                out_channels=nout,
                kernel_size=1,
                stride=1,
                bias=False,
                dimension=self._D),
            ME.MinkowskiBatchNorm(nout),
            ME.MinkowskiReLU()
        )
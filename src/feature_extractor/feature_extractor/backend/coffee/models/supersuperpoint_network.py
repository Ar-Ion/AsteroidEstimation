import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

class SuperPoint(ME.MinkowskiNetwork):

    def __init__(self, in_feat=1, out_feat=256, D=2):
        super(SuperPoint, self).__init__(D)

        self._D = D

        self.conv1a = self.create_conv(1, 64)
        self.conv1b = self.create_downsampling_conv(64, 64)
        self.conv2a = self.create_conv(64, 64)
        self.conv2b = self.create_downsampling_conv(64, 64)
        self.conv3a = self.create_conv(64, 128)
        self.conv3b = self.create_downsampling_conv(128, 128)
        self.conv4a = self.create_conv(128, 128)
        self.conv4b = self.create_conv(128, 128)
        self.convDa = self.create_conv(128, 256)
        self.convDb = self.create_mapping_conv(256, 256)
        self.convD1 = self.create_upsampling_conv(256, 256)
        self.convD2 = self.create_upsampling_conv(256, 256)
        self.convD3 = self.create_upsampling_conv(256, 256)

        self.prune1 = ME.MinkowskiPruning()
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def forward(self, x):
        out = self.conv1a(x)
        out = self.conv1b(out)
        out = self.conv2a(out)
        out = self.conv2b(out)
        out = self.conv3a(out)
        out = self.conv3b(out)
        out = self.conv4a(out)
        out = self.conv4b(out)
        
        out = self.convDa(out)
        out = self.convDb(out)
        out = self.convD1(out)
        out = self.convD2(out)
        out = self.convD3(out)
        
        mask = torch.rand(out.features.size(0)) < 4096/out.features.size(0)
        out = self.prune1(out, mask)
        
        return out

    def create_downsampling_layer(self, nin, nout, count):
        layer = []

        for i in range(count):
            layer.append(self.create_conv(nin, nout))

        layer.append(self.create_downsampling_conv(nin, nout))

        return nn.Sequential(*layer)
    
    def create_upsampling_layer(self, nin, nout, count):
        layer = []

        for i in range(count):
            layer.append(self.create_conv(nin, nout))

        layer.append(self.create_downsampling_conv(nin, nout))

        return nn.Sequential(*layer)
    
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
            ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=self._D)
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
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

class SparseSuperPoint(ME.MinkowskiNetwork):

    def __init__(self, D=2):
        super(SparseSuperPoint, self).__init__(D)

        self._D = D

        self._conv_kernel = ME.KernelGenerator(
            kernel_size=3,
            stride=1,
            dilation=1,
            region_type=ME.RegionType.HYPER_CUBE,
            dimension=D
        )

        self._down_kernel = ME.KernelGenerator(
            kernel_size=3,
            stride=1,
            dilation=1,
            region_type=ME.RegionType.HYPER_CUBE,
            dimension=D
        )

        self._up_kernel = ME.KernelGenerator(
            kernel_size=3,
            stride=1,
            dilation=1,
            region_type=ME.RegionType.HYPER_CUBE,
            dimension=D
        )

        self.conv0 = self.create_conv(1, 32)

        self.conv1a = self.create_convs(32, 1)
        self.conv1b = self.create_downsampling_conv(32, 64)

        self.conv2a = self.create_convs(64, 1)
        self.conv2b = self.create_downsampling_conv(64, 128)

        self.conv3a = self.create_convs(128, 1)
        self.conv3b = self.create_downsampling_conv(128, 256)
        
        #self.conv4 = self.create_upsampling_conv(256, 256)
        #self.conv5 = self.create_upsampling_conv(256+128, 256+128)
        #self.conv6 = self.create_upsampling_conv(256+128+64, 256+128+64)
                
        self.mlp = self.create_mlp(256, 128)

        self.convMap = self.create_final_conv(128, 128)

        self.interp = ME.MinkowskiInterpolation()

    def forward(self, x):

        out = self.conv0(x)
        out_s1 = self.conv1a(out)
        out = self.conv1b(out_s1)
        out_s2 = self.conv2a(out)
        out = self.conv2b(out_s2)
        out_s4 = self.conv3a(out)
        out = self.conv3b(out_s4)
        
        #out = self.conv4(out)
        #out = ME.cat(out, out_s4)
        
        #out = self.conv5(out)
        #out = ME.cat(out, out_s2)
        
        #out = self.conv6(out)
        #out = ME.cat(out, out_s1)
        
        out = self.mlp(out)
        out = self.convMap(out)

        out_coords = x.coordinates
        out_features = self.interp(out, out_coords.to(x.features.dtype))

        out = ME.SparseTensor(out_features.contiguous(), out_coords)

        return MF.normalize(out, dim=1).features

    def create_mlp(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiLinear(nin, nout, bias=False),
            ME.MinkowskiInstanceNorm(nout),
            ME.MinkowskiReLU()
        )
        
    def create_convs(self, n, count):
        convs = []

        for i in range(count):
            convs.append(self.create_conv(n, n))

        return nn.Sequential(*convs)

    def create_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=nin,
                out_channels=nout,
                kernel_generator=self._conv_kernel,
                dimension=self._D,
                bias=False
            ),
            ME.MinkowskiInstanceNorm(nout),
            ME.MinkowskiReLU()
        )
    
    def create_downsampling_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=nin,
                out_channels=nout,
                kernel_generator=self._down_kernel,
                dimension=self._D,
                bias=False
            ),
            ME.MinkowskiMaxPooling(
                kernel_size=2,
                stride=2,
                dimension=self._D
            ),
            ME.MinkowskiInstanceNorm(nout),
            ME.MinkowskiReLU()
        )
    
    def create_upsampling_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=nin,
                out_channels=nout,
                kernel_generator=self._up_kernel,
                dimension=self._D,
                bias=False
            ),
            ME.MinkowskiPoolingTranspose(
                kernel_size=2, 
                stride=2, 
                dimension=self._D
            ),
            ME.MinkowskiInstanceNorm(nout),
            ME.MinkowskiReLU()
        )
    
    def create_final_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=nin,
                out_channels=nout,
                kernel_generator=self._conv_kernel,
                dimension=self._D,
                bias=True
            ),
            ME.MinkowskiReLU()
        )

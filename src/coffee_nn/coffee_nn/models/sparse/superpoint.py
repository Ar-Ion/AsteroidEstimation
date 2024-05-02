import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

class SuperPoint(ME.MinkowskiNetwork):

    def __init__(self, D=2):
        super(SuperPoint, self).__init__(D)

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
            dilation=3,
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

        self.conv0 = self.create_conv(1, 64)

        self.conv1a = self.create_convs(64, 2)
        self.conv1b = self.create_downsampling_conv(64, 128)

        self.conv2a = self.create_convs(128, 2)
        self.conv2b = self.create_downsampling_conv(128, 256)

        self.conv3a = self.create_convs(256, 2)
        self.conv3b = self.create_downsampling_conv(256, 256)
        self.conv3c = self.create_mapping_conv(256, 256)

        self.conv4 = self.create_upsampling_conv(256, 256)
        self.conv5 = self.create_upsampling_conv(256+256, 256+256)
        self.conv6 = self.create_upsampling_conv(256+256+128, 256+256+128)

        self.interp = ME.MinkowskiInterpolation()

    def forward(self, x):
        out = self.conv0(x)
        out_s1 = self.conv1a(out)
        out = self.conv1b(out_s1)
        out_s2 = self.conv2a(out)
        out = self.conv2b(out_s2)
        out_s4 = self.conv3a(out)
        out = self.conv3b(out_s4)
        out = self.conv3c(out)
        
        out = self.conv4(out)
        out = ME.cat(out, out_s4)
        
        out = self.conv5(out)
        out = ME.cat(out, out_s2)
        
        out = self.conv6(out)
        out = ME.cat(out, out_s1)

        return MF.normalize(MF.relu(out), dim=1)

        #out_coords = x.coordinates
        #out_features = self.interp(out, out_coords.to(torch.float))
                
        #return ME.SparseTensor(out_features, out_coords)

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
                dimension=self._D
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiBatchNorm(nout)
        )
    
    def create_downsampling_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=nin,
                out_channels=nout,
                kernel_generator=self._down_kernel,
                dimension=self._D
            ),
            ME.MinkowskiMaxPooling(
                kernel_size=2,
                stride=2,
                dimension=self._D
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiBatchNorm(nout)
        )
    
    def create_upsampling_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=nin,
                out_channels=nout,
                kernel_generator=self._up_kernel,
                dimension=self._D
            ),
            ME.MinkowskiPoolingTranspose(
                kernel_size=2, 
                stride=2, 
                dimension=self._D
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiBatchNorm(nout)
        )
    
    def create_mapping_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=nin,
                out_channels=nout,
                kernel_generator=self._conv_kernel,
                dimension=self._D
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiBatchNorm(nout)
        )
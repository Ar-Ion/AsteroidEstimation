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

        self.conv4 = self.create_upsampling_conv(256, 256)
        self.conv5 = self.create_upsampling_conv(256+128, 256+128)
        self.conv6 = self.create_upsampling_conv(256+128+64, 256+128+64)
                
        self.mlp = self.create_mlp(256+128+64+32, 128)

        self.convMap = self.create_mapping_conv(128, 128)

        self.interp = ME.MinkowskiInterpolation()

    def forward(self, x):
        
        #x = MF.normalize(MF.relu(x))

        pi_2 = 3.141592/2

        in_coords = x.coordinates
        in_features = x.features/pi_2 - 0.5
        input = ME.SparseTensor(in_features, in_coords)

        out = self.conv0(input)
        out_s1 = self.conv1a(out)
        out = self.conv1b(out_s1)
        out_s2 = self.conv2a(out)
        out = self.conv2b(out_s2)
        out_s4 = self.conv3a(out)
        out = out_s4#self.conv3b(out_s4)
        
        # out = self.conv4(out)
        # out = ME.cat(out, out_s4)
        
        # out = self.conv5(out)
        # out = ME.cat(out, out_s2)
        
        # out = self.conv6(out)
        # out = ME.cat(out, out_s1)
        
        # out = self.mlp(out)
        # out = self.convMap(out)

        out_coords = x.coordinates
        out_features = self.interp(out, out_coords.to(input.features.dtype))
                
        out = ME.SparseTensor(out_features, out_coords)
        
        return MF.normalize(MF.relu(out), dim=1)

    def create_mlp(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiLinear(nin, nout, bias=False),
            ME.MinkowskiBatchNorm(nout),
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
                dimension=self._D,
                bias=True
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiBatchNorm(nout)
        )
    
# class SparseSuperPoint(ME.MinkowskiNetwork):

#     def __init__(self, D=2):
#         super(SparseSuperPoint, self).__init__(D)

#         self._D = D

#         self._conv_kernel = ME.KernelGenerator(
#             kernel_size=3,
#             stride=1,
#             dilation=1,
#             region_type=ME.RegionType.HYPER_CUBE,
#             dimension=D
#         )

#         self._down_kernel = ME.KernelGenerator(
#             kernel_size=3,
#             stride=1,
#             dilation=3,
#             region_type=ME.RegionType.HYPER_CUBE,
#             dimension=D
#         )

#         self._up_kernel = ME.KernelGenerator(
#             kernel_size=3,
#             stride=1,
#             dilation=1,
#             region_type=ME.RegionType.HYPER_CUBE,
#             dimension=D
#         )

#         self.conv1a = self.create_conv(1, 32)
#         self.conv1b = self.create_downsampling_conv(32, 64)

#         self.conv2a = self.create_conv(64, 64)
#         self.conv2b = self.create_downsampling_conv(64, 128)

#         self.conv3a = self.create_conv(128, 128)
#         self.conv3b = self.create_downsampling_conv(128, 128)

#         self.conv4a = self.create_conv(128, 128)
#         self.conv4b = self.create_downsampling_conv(128, 256)

#         self.convPa = self.create_conv(256, 256)
#         self.convPb = self.create_conv(256, 65)

#         self.convDa = self.create_conv(256, 256)
#         self.convDb = self.create_conv(256, 256)

#         self.interp = ME.MinkowskiInterpolation()

#     def forward(self, x):
        
#         #x = MF.normalize(MF.relu(x))

#         pi_2 = 3.141592/2

#         in_coords = x.coordinates
#         in_features = x.features/pi_2 - 0.5
#         input = ME.SparseTensor(in_features, in_coords)

#         out = self.conv1a(input)
#         out = self.conv1b(out)
#         out = self.conv2a(out)
#         out = self.conv2b(out)
#         out = self.conv3a(out)
#         out = self.conv3b(out)
#         out = self.conv4a(out)
#         out = self.conv4b(out)
        
#         outP = self.convPa(out)
#         outP = self.convPb(outP)

#         outP_coords = x.coordinates
#         outP_features = self.interp(out, outP_coords.to(x.features.dtype))[:, :-1]
#         outP = MF.softmax(ME.SparseTensor(outP_features, outP_coords), dim=1)
        
#         outD = self.convDa(out)
#         outD = self.convDb(outD)

#         outD_coords = x.coordinates
#         outD_features = self.interp(out, outD_coords.to(x.features.dtype)) 
#         outD = MF.normalize(MF.relu(ME.SparseTensor(outD_features, outD_coords)), dim=1)

#         return (outP, outD)
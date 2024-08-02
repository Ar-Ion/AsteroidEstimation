import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

class SparseSuperPoint(ME.MinkowskiNetwork):
    
    BLOCK = Bottleneck
    LAYERS = (1, 1, 1, 1)
    PLANES = (8, 16, 32, 64)

    def __init__(self, D=2):
        super(SparseSuperPoint, self).__init__(D)
        
        self._D = D
        self.inplanes = self.PLANES[0]
        
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                1, self.inplanes, kernel_size=3, stride=1, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )

        self.conv5 = self.create_upsampling_conv(self.inplanes, 256)
        self.conv6 = self.create_upsampling_conv(256, 256)
        self.conv7 = self.create_upsampling_conv(256, 256)
        self.conv8 = self.create_upsampling_conv(256, 256)
        
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        
        return MF.normalize(out, dim=1)
    
    def create_downsampling_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=nin,
                out_channels=nout,
                stride=2,
                kernel_size=3,
                dimension=self._D,
                bias=False
            ),
            ME.MinkowskiBatchNorm(nout),
            ME.MinkowskiReLU()
        )
    
    def create_upsampling_conv(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=nin,
                out_channels=nout,
                stride=2,
                kernel_size=3,
                dimension=self._D,
                bias=False
            ),
            ME.MinkowskiBatchNorm(nout),
            ME.MinkowskiReLU()
        )
        
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)
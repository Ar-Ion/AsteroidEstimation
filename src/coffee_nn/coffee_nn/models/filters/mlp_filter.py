import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

class SparseFilter(ME.MinkowskiNetwork):

    def __init__(self, D=2):
        super(SparseFilter, self).__init__(D)

        self._D = D
                
        self.mlp1 = self.create_mlp(3, 64)
        self.mlp2 = self.create_mlps(64, 8)
        self.mlp3 = nn.Sequential( # Third layer without activation (handled by the loss function)
            ME.MinkowskiLinear(64, 1, bias=True),
            ME.MinkowskiBatchNorm(1)
        )
        
    def forward(self, x):

        normalization = torch.tensor((1024, 1024, 3.141592/2))
        features = torch.hstack((x.coordinates[:, 1, None], x.coordinates[:, 2, None], x.features))
    
        in_coords = x.coordinates
        in_features = features/normalization - 0.5
        input = ME.SparseTensor(in_features, in_coords)

        out = self.mlp1(input)
        out = self.mlp2(out)
        out = self.mlp3(out)

        return out

    def create_mlps(self, n, count):
        convs = []

        for i in range(count):
            convs.append(self.create_mlp(n, n))

        return nn.Sequential(*convs)
    
    def create_mlp(self, nin, nout):
        return nn.Sequential(
            ME.MinkowskiLinear(nin, nout, bias=True),
            ME.MinkowskiBatchNorm(nout),
            ME.MinkowskiReLU()
        )
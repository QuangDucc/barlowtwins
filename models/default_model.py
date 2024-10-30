from utils.modules import *

activation = nn.LeakyReLU(0.2, inplace=True)


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, groups=1, act=True):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel, stride, padding, groups=groups)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

        if act:
            self.act = activation


class DeconvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.decon = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, output_padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = activation

    def forward(self, input):
        results = super().forward(input)
        return results[..., 1:-1, 1:-1, 1:-1]


class ResBlock(nn.Module):

    def __init__(self, channels, embedding, groups=16, ratio=1/4):
        super().__init__()

        self.residual = nn.Sequential(*[
            ConvBlock(channels, embedding * groups, kernel=1, padding=0),
            ConvBlock(embedding * groups, embedding * groups, groups=groups),
            ConvBlock(embedding * groups, channels, kernel=1, padding=0, act=False),
            SEAttention(channels, ratio, dims=[-1, -2, -3]),
        ])

        self.act = activation

    def forward(self, x):
        res = self.residual(x)
        res = x + res

        return self.act(res)


class UpBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, embeddings, groups):
        super().__init__()

        self.decon = DeconvBlock(in_channels, out_channels)

        self.reduce = nn.Sequential(*[
            ConvBlock(out_channels + skip_channels, out_channels),
            ResBlock(out_channels, embeddings, groups)
        ])

    def forward(self, x, skip):
        up = self.decon(x)
        combine = torch.cat([up, skip], dim=1)

        return self.reduce(combine)


class Segmentator(nn.Module):

    def __init__(self, in_channels=1, stem=32, encoders=(64, 160, 352), embeddings=4, groups=16, n_class=2):
        super().__init__()

        self.stem = ConvBlock(in_channels, stem, kernel=7, padding=3, stride=2)

        downs = []
        prev = stem
        for e in encoders:
            downs.append(nn.Sequential(*[
                ConvBlock(prev, e, stride=2),
                ResBlock(e, embeddings, groups),
                ResBlock(e, embeddings, groups)
            ]))

            prev = e
            embeddings *= 2

        self.downs = nn.ModuleList(downs)

        ups = []
        for d in [*encoders[::-1][1:], stem]:
            embeddings = embeddings // 2
            ups.append(UpBlock(prev, d, d, embeddings, groups))
            prev = d

        self.ups = nn.ModuleList(ups)

    def forward(self, x):
        stem = self.stem(x)

        prev = stem
        skips = []
        for d in self.downs:
            skips.append(prev)
            prev = d(prev)
        
        skips = skips[::-1]
        outs = [prev]
        for skip, up in zip(skips, self.ups):
            prev = up(prev, skip)
            outs.append(prev)

        return outs

def segmentator(**kwargs):
    return Segmentator(**kwargs)
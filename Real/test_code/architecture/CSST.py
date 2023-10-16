import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
import numpy as np

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type:(Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn      #HS_MSA or FFN
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class SS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=28,
            heads=8,                          # heads = dim_scale//dim  1,2
            only_local_branch=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads                    # 1,2
        self.scale = dim_head ** -0.5         # 1/sqrt(28) = 0.189
        self.window_size = window_size        # (8,8)
        self.only_local_branch = only_local_branch

        # position embedding
        if only_local_branch:
            seq_l = window_size[0] * window_size[1]       # 64
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l*4, seq_l*4))            #(1,1,64,64)
            trunc_normal_(self.pos_emb)                                                      #trunc_normal_(parameter(1,1,64,64))
        else:
            seq_l1 = window_size[0] * window_size[1]       # 64
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l1, seq_l1))       #(1,1,0,64,64);(1,1,1,64,64)
            # h,w = 256//self.heads,320//self.heads
            h, w = 256 // self.heads, 256 // self.heads                                      #256,256;    128,128
            seq_l2 = h*w//seq_l1                                                             #256*256//64=1024; 128*128//64=256
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l2, seq_l2))       #(1,1,1,64,64)
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads  #28, 56
        # x:b,h,w,28;
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape                                                # 5 256 256 28;   5 256 256 28
        w_size = self.window_size                                           #(8,8)
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'

        if self.only_local_branch:
            #x: 5 256 256 28
            q = self.to_q(x)                                                # 5 256 256 28
            k, v = self.to_kv(x).chunk(2, dim = -1)                         # 5 256 256 28
            # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))  # [5120,1,64,28]
            q, k, v = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0]*2, b1=w_size[1]*2), (q, k, v))          # 256=16*16 [5,256,256,28]
            q, k, v = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=1),
                             (q, k, v))  # [5,256,1,256,28]
            q *= self.scale
            sim = einsum('b n h i d, b n h j d -> b n h i j', q, k)                                  # [5,256,1,256,256]
            sim = sim + self.pos_emb
            #
            sim = torch.roll(sim, shifts=(1, 1), dims=(3, 4))
            attn = sim.softmax(dim=-1)  # [5,256,1,256,256]
            out = einsum('b n h i j, b n h j d -> b n h i d', attn, v)       # [5,256,1,256,28]
            out = rearrange(out, 'b n h mm d -> b n mm (h d)')               # [5,256,256,28]

        else:
            q = self.to_q(x)                                                 # 5 128 128 56 heads=2
            k, v = self.to_kv(x).chunk(2, dim=-1)                            # [5,128,128,56]
            q1, q2 = q[:,:,:,:c//2], q[:,:,:,c//2:]
            k1, k2 = k[:,:,:,:c//2], k[:,:,:,c//2:]
            v1, v2 = v[:,:,:,:c//2], v[:,:,:,c//2:]                          # [5,128,128,28]

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                              b0=w_size[0], b1=w_size[1]), (q1, k1, v1))      # 128=16*8 [5,256,64,28]
            #对传入的参数t执行rearrane:[b,8h,8w,c]->[b,hw,64,c]操作
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2),
                             (q1, k1, v1))                                                    # [5,256,1,64,28]
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)                        # [5,256,1,64,64]
            sim1 = sim1 + self.pos_emb1                                                       # [5,256,1,64,64]
            # fixed shifts
            # shifts = np.random.choice((2, -2), 2)
            # sim1 = torch.roll(sim1, shifts=(int(shifts[0]),int(shifts[1])), dims=(3, 4))
            sim1 = torch.roll(sim1, shifts=(1, 1), dims=(3, 4))
            attn1 = sim1.softmax(dim=-1) #[5,256,1,64,64]
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)                     # [5,256,1,64,28]
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')                              # [5,256,64,28]

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))          # 128=16*8 [5,256,64,28]
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))  # [5,64,256,28]
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2),
                             (q2, k2, v2))                                                           # [5,64,1,256,28]
            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            sim2 = sim2 + self.pos_emb2                                                              # [5,64,1,256,256]
            # fixed shifts
            # shifts = np.random.choice((2, -2), 2)
            # sim2 = torch.roll(sim2, shifts=(int(shifts[0]),int(shifts[1])), dims=(3, 4))
            sim2 = torch.roll(sim2, shifts=(1, 1), dims=(3, 4))
            attn2 = sim2.softmax(dim=-1)   # [5,64,1,256,256]
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)  # [5,64,1,256,28]
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')           # [5,64,256,28]
            out2 = out2.permute(0, 2, 1, 3)                                # [5,256,64,28]

            out = torch.cat([out1,out2],dim=-1).contiguous()               # [5,256,64,56]
            out = self.to_out(out)                                         # [5,256,64,56]
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])                                  # [5,128,128,56]
        return out

class SSAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            num_blocks=2,    # input as num_blocks[i], num_blocks=[1,1,1]
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, SS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, only_local_branch=(heads==1))),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)    #[b,h,w,c]
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class SST(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, num_blocks=[1,1,1]):
        super(SST, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                SSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
            ]))
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = SSAB(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                SSAB(dim=dim_scale // 2, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim,
                     heads=(dim_scale // 2) // dim),
            ]))
            dim_scale //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb   # 0
        pad_w = (wb - w_inp % wb) % wb   # 0
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Embedding
        fea = self.embedding(x)   #[b,28,256,256]
        x = x[:,:28,:,:]          #[b,28,256,256]

        # Encoder
        fea_encoder = []
        for (SSAB, FeaDownSample) in self.encoder_layers:
            fea = SSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, SSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales-2-i]], dim=1))
            fea = SSAB(fea)

        # Mapping
        out = self.mapping(fea) + x
        return out[:, :, :h_inp, :w_inp]

class QECNet(nn.Module):
    def __init__(self, in_nc=28, out_nc=5, channel=64):   #out_nc = num*2  --> out_nc = num
        super(QECNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):
        x = self.down_sample(self.relu(self.fution(x)))         # [b,28,256,256]-->[b,64,256,256]-->[b,64,256,256]-->[b,64,256,256]
        x = self.mlp(x) + 1e-6                                  # [b,64,256,256]-->[b,5,256,256]
        return x[:, :, :, :]

class CSST(nn.Module):   
    def __init__(self, num_iterations=1):                       # num_iterations = 5
        super(CSST, self).__init__()
        self.para_estimator = QECNet(in_nc=28, out_nc=num_iterations)
        self.upsize = nn.ConvTranspose2d(4, 4, 2, stride=2, padding=0, output_padding=0)
        self.up = nn.ConvTranspose2d(3, 3, 2, stride=10, padding=8, output_padding=0)
        self.down1 = nn.Conv2d(28, 3, 1, stride=1, padding=0)
        self.down2 = nn.Conv2d(28, 3, 2, stride=2, padding=0)
        self.fution = nn.Conv2d(7, 28, 1, padding=0, bias=True)
        self.num_iterations = num_iterations
        self.denoisers = nn.ModuleList([])
        for _ in range(num_iterations):
            self.denoisers.append(
                SST(in_dim=35, out_dim=28, dim=28, num_blocks=[1,1,1]),
            )

    def initial(self, y_h, y_l, pattern):
        # y_h = self.upsize(y_h)             #4
        y_l = self.down1(y_l)                 #3
        pattern = self.down2(pattern)       #3
        z = self.fution(torch.cat([y_h, y_l, pattern], dim=1))                               #[b,6,256,256] --> [b,28,256,256]
        beta = self.para_estimator(self.fution(torch.cat([y_h, y_l, pattern], dim=1)))       #[b,28,256,256] --> [b,5,1,1] 其实每个阶段的迭代参数
        return z, beta, y_l, pattern

    def forward(self, y_h, y_l, pattern):
        """
        :param y: [b,1, 256,256]
        :param y_l: [b,28,256,256] x
        :param pattern: [b,28,512,512]
        """
        z, betas, y_l, pattern = self.initial(y_h, y_l, pattern)
        z = z.contiguous()
        betas = betas.contiguous()
        for i in range(self.num_iterations):
            beta = betas[:, i:i+1, :, :]                                       # [b,1,256,256]
            x = z
            z = self.denoisers[i](torch.cat([x, beta, y_l, pattern],dim=1))    # [b, 32, 256, 256]
        return z[:, :, :, :]

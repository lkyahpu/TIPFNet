import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
from function import DenseBlock_light, ConvLayer, Fusion_network, Dense_encoder
from matplotlib import pyplot as plt


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0].type(torch.long), self.relative_indices[:, :, 1].type(torch.long)]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 4, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )
        # self.dense_encoder=Dense_encoder()

    def forward(self, img):
        en_x1 = self.stage1(img)
        en_x2 = self.stage2(en_x1)
        en_x3 = self.stage3(en_x2)
        en_x4 = self.stage4(en_x3)

        en = [en_x1, en_x2, en_x3, en_x4]

        return en


class swin_fusion_net(nn.Module):
    def __init__(self, nb_filter=[32, 64, 128, 128], f_type='res'):
        super().__init__()

        self.fusion_model = Fusion_network(nb_filter, f_type)
        block = DenseBlock_light
        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], 3, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], 3, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], 3, 1)
        # short connection
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], 3, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], 3, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], 3, 1)

        self.up = nn.Upsample(scale_factor=2)

        self.conv_out = ConvLayer(nb_filter[0], 1, 1, 1)
        self.swin_t = SwinTransformer(
            hidden_dim=32,  # 96
            layers=(2, 2, 2, 2),  # 2,2,6,2
            heads=(3, 3, 3, 3),  # 3, 6, 12, 24
            channels=1,
            num_classes=3,
            head_dim=32,
            window_size=7,
            downscaling_factors=(1, 2, 2, 2),
            relative_pos_embedding=True
        )
        self.dense_encoder = Dense_encoder()
        self.conv_f0 = ConvLayer(64, 32, 1, 1)
        self.conv_f1 = ConvLayer(128, 64, 1, 1)
        self.conv_f2 = ConvLayer(256, 128, 1, 1)
        self.conv_f3 = ConvLayer(256, 128, 1, 1)


    def forward(self, x, y):

        en_ir = self.swin_t(x)
        en_vi = self.swin_t(y)

        # en_ir = self.dense_encoder(x)
        # en_vi = self.dense_encoder(y)

        # out0 = torch.cat([en_ir[0], en_vi[0]], 1)
        # out1 = torch.cat([en_ir[1], en_vi[1]], 1)
        # out2 = torch.cat([en_ir[2], en_vi[2]], 1)
        # out3 = torch.cat([en_ir[3], en_vi[3]], 1)
        #
        # f_en0 = self.conv_f0(out0)
        # f_en1 = self.conv_f1(out1)
        # f_en2 = self.conv_f2(out2)
        # f_en3 = self.conv_f3(out3)
        #
        # f_en = [f_en0,f_en1,f_en2,f_en3]

        f_en = self.fusion_model(en_ir, en_vi)

        # out = f_en[3].detach().cpu().numpy().squeeze()
        #
        # for i in range(16):
        #     plt.subplot(4, 4, i + 1)
        #     plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
        #     plt.imshow(out[i, :, :])
        #     plt.axis('off')
        # plt.show()

        # plt.imshow(out[5, :, :])
        # plt.show()

        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        output = self.conv_out(x1_3)
        en_output = self.swin_t(output)


        return output, en_output, f_en

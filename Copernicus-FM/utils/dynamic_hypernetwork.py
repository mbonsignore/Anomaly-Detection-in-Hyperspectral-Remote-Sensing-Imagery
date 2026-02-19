import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .flexivit.patch_embed import pi_resize_patch_embed
from .flexivit.utils import to_2tuple
from .aurora.fourier import FourierExpansion


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(transformer_output[-1])
        return weights, bias


class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(conv)
        if not bias:
            self.conv.add_module("ln", nn.LayerNorm(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        return x + y


class Dynamic_MLP_Decoder(nn.Module):
    def __init__(self, wv_planes, inter_dim=128, kernel_size=16, decoder_embed=512):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.inter_dim = inter_dim
        self.decoder_embed = decoder_embed
        self._num_kernel = self.kernel_size * self.kernel_size * self.decoder_embed

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, decoder_embed
        )
        self.scaler = 0.01

        self._init_weights()

    def _get_weights(self, waves, batch=True):
        if batch:
            return self.weight_generator(waves)
        dweights = []
        for i in range(waves.size(0)):
            dweights.append(self.weight_generator(waves[i]))
        return torch.stack(dweights, dim=0)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        self.weight_generator.apply(self.weight_init)

    def forward(self, img_feat, waves, kernel_size=None):
        inplanes = waves.size(0)
        weight, bias = self._get_weights(waves)

        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.decoder_embed)
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])
        if kernel_size is not None and self.kernel_size != kernel_size:
            dynamic_weight = pi_resize_patch_embed(dynamic_weight, (kernel_size, kernel_size))
        else:
            kernel_size = self.kernel_size
        dynamic_weight = dynamic_weight.permute([1, 2, 3, 0]).contiguous().view(-1, self.decoder_embed)

        weights = dynamic_weight * self.scaler
        dynamic_out = F.linear(img_feat, weights, bias=None)
        return dynamic_out


class Dynamic_Patch_Embed(nn.Module):
    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.patch_size = to_2tuple(kernel_size)
        self.weight2 = nn.Parameter(torch.empty([embed_dim, 2, kernel_size, kernel_size]))
        self.bias2 = nn.Parameter(torch.empty([embed_dim]))
        self.weight3 = nn.Parameter(torch.empty([embed_dim, 3, kernel_size, kernel_size]))
        self.bias3 = nn.Parameter(torch.empty([embed_dim]))
        self.weight4 = nn.Parameter(torch.empty([embed_dim, 4, kernel_size, kernel_size]))
        self.bias4 = nn.Parameter(torch.empty([embed_dim]))
        self.weight9 = nn.Parameter(torch.empty([embed_dim, 9, kernel_size, kernel_size]))
        self.bias9 = nn.Parameter(torch.empty([embed_dim]))
        self.weight70 = nn.Parameter(torch.empty([embed_dim, 70, kernel_size, kernel_size]))
        self.bias70 = nn.Parameter(torch.empty([embed_dim]))
        self.weights = {2: self.weight2, 3: self.weight3, 4: self.weight4, 9: self.weight9, 70: self.weight70}
        self.biass = {2: self.bias2, 3: self.bias3, 4: self.bias4, 9: self.bias9, 70: self.bias70}

    def forward(self, img_feat, waves):
        inplanes = waves.size(0)
        weights = self.weights[inplanes]
        bias = self.biass[inplanes]
        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )
        x = dynamic_out.flatten(2).transpose(1, 2)
        return x


class Dynamic_MLP_OFA(nn.Module):
    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = to_2tuple(kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01
        self.fclayer = FCResLayer(wv_planes)
        self._init_weights()

    def _get_weights(self, waves):
        return self.weight_generator(waves)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs, kernel_size=None):
        inplanes = wvs.size(0)
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)

        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if kernel_size is not None and self.kernel_size != kernel_size:
            dynamic_weight = pi_resize_patch_embed(dynamic_weight, (kernel_size, kernel_size))
        else:
            kernel_size = self.kernel_size

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler
        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=kernel_size, padding=1, dilation=1
        )

        x = dynamic_out.flatten(2).transpose(1, 2)
        return x, waves


class Dynamic_MLP_OFA_spectral(nn.Module):
    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = to_2tuple(kernel_size)
        self.num_patches = -1

        self.spectrum_central_expansion = FourierExpansion(100, 1e9)
        self.spectrum_bandwidth_expansion = FourierExpansion(1, 1e9)

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01
        self.fclayer = FCResLayer(wv_planes)
        self._init_weights()

    def _get_weights(self, waves):
        return self.weight_generator(waves)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs, bandwidths, kernel_size=None):
        inplanes = wvs.size(0)
        emb_central = self.spectrum_central_expansion(wvs, self.wv_planes)
        emb_bandwidth = self.spectrum_bandwidth_expansion(bandwidths, self.wv_planes)
        waves = emb_central + emb_bandwidth
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)

        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if kernel_size is not None and self.kernel_size != kernel_size:
            dynamic_weight = pi_resize_patch_embed(dynamic_weight, (kernel_size, kernel_size))
        else:
            kernel_size = self.kernel_size

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler
        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=kernel_size, padding=1, dilation=1
        )

        x = dynamic_out.flatten(2).transpose(1, 2)
        return x, waves


class Dynamic_MLP_OFA_variable(nn.Module):
    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = to_2tuple(kernel_size)
        self.num_patches = -1

        self.language_proj = nn.Linear(2048, self.wv_planes)
        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01
        self.fclayer = FCResLayer(wv_planes)
        self._init_weights()

    def _get_weights(self, waves):
        return self.weight_generator(waves)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, language_embed, kernel_size=None):
        emb_language = language_embed.unsqueeze(0)
        waves = self.language_proj(emb_language)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)

        inplanes = waves.size(0)
        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if kernel_size is not None and self.kernel_size != kernel_size:
            dynamic_weight = pi_resize_patch_embed(dynamic_weight, (kernel_size, kernel_size))
        else:
            kernel_size = self.kernel_size

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler
        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=kernel_size, padding=1, dilation=1
        )

        x = dynamic_out.flatten(2).transpose(1, 2)
        return x, waves


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

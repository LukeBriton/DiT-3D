import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.voxelization import Voxelization
import modules.functional as voxel_ops


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings, labels


#################################################################################
#                               Norm / Blocks                                   #
#################################################################################

class LayerNorm3d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return F.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}; group={self.num_groups}"


class FinalLayer3d(nn.Module):
    """
    The final layer of DiC-3D-Conv.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm3d(hidden_size, affine=False, eps=1e-6)
        self.out_proj = nn.Conv3d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.out_proj(x)
        return x


class OverlapPatchEmbed3d(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample3d(nn.Module):
    def __init__(self, n_feat, out_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(n_feat, out_feat, kernel_size=3, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x)


class Upsample3d(nn.Module):
    def __init__(self, n_feat, out_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(n_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x)


class UNetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels=None,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        blockconfig=0,
        actfunc="silu",
        actf=(1, 1),
        affinef=2,
        norm_type="gnorm",
        norm_type1="gnorm",
        affine=1,
        actinada=0,
        init_zero=0,
        use_gamma=False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels

        self.dropout = dropout
        self.skip_scale = skip_scale
        self.blockconfig = blockconfig
        self.affinef = affinef
        self.norm_type = norm_type
        self.norm_type1 = norm_type1
        self.layernorm_affine = affine
        self.actinada = actinada
        self.init_zero = init_zero

        if norm_type == "gnorm":
            self.norm0 = GroupNorm(
                num_channels=in_channels,
                eps=eps,
                num_groups=kwargs.get("num_groups", 32),
                min_channels_per_group=kwargs.get("min_channels", 4),
            )

        self.conv0 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.affine = nn.Sequential(
            nn.SiLU() if actinada else nn.Identity(),
            nn.Linear(
                in_features=emb_channels,
                out_features=(in_channels if self.blockconfig else out_channels) * self.affinef,
                bias=True,
            ),
        )

        if self.init_zero:
            nn.init.constant_(self.affine[-1].weight, 0)
            nn.init.constant_(self.affine[-1].bias, 0)

        if norm_type1 == "gnorm":
            self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)

        self.conv1 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.ones(out_channels))

        self.skip = None
        self.act = nn.GELU if actfunc == "gelu" else nn.SiLU
        self.act0 = self.act() if actf[0] else nn.Identity()
        self.act1 = self.act() if actf[1] else nn.Identity()
        self.use_gamma = use_gamma

        if out_channels != in_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, emb):
        if self.blockconfig == 0:
            orig = x
            x = self.conv0(self.act0(self.norm0(x)))

            params = self.affine(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
            if self.affinef == 2:
                scale, shift = params.chunk(chunks=2, dim=1)
                gate = 1
            elif self.affinef == 3:
                gate, scale, shift = params.chunk(chunks=3, dim=1)
            x = self.act1(torch.addcmul(shift, self.norm1(x), scale + 1))

            x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
            if self.use_gamma:
                x = x * self.gamma.view(1, -1, 1, 1, 1)

            x = (gate * x).add_(self.skip(orig) if self.skip is not None else orig)
            x = x * self.skip_scale

        elif self.blockconfig == 1:
            orig = x

            params = self.affine(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
            if self.affinef == 2:
                scale, shift = params.chunk(chunks=2, dim=1)
                gate = 1
            elif self.affinef == 3:
                gate, scale, shift = params.chunk(chunks=3, dim=1)
            x = self.conv0(self.act0(torch.addcmul(shift, self.norm0(x), scale + 1)))

            x = self.act1(self.norm1(x))

            x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
            if self.use_gamma:
                x = x * self.gamma.view(1, -1, 1, 1, 1)
            x = (gate * x).add_(self.skip(orig) if self.skip is not None else orig)
            x = x * self.skip_scale
        else:
            raise NotImplementedError()
        return x


class U_Block(nn.Module):
    def __init__(self, input_size, hidden_size, input_chans=None, **kwargs):
        super().__init__()
        self.conv = UNetBlock3d(
            input_chans if input_chans else hidden_size,
            hidden_size,
            emb_channels=hidden_size,
            **kwargs,
        )

    def forward(self, x, c):
        return self.conv(x, c)


#################################################################################
#                           Positional Embeddings                               #
#################################################################################

def get_3d_sincos_pos_embed(embed_dim, grid_size):
    grid = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid, grid, grid, indexing="ij")
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    return get_3d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                                 DiC-3D Conv                                   #
#################################################################################

class DiC3DConv(nn.Module):
    """
    3D DiC with Conv3d backbone and optional 3D sin-cos positional channels.
    """
    def __init__(
        self,
        input_size=32,
        in_channels=4,
        hidden_size=1152,
        depth=(4, 10, 16, 10, 4),
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        rep=1,
        ffn_type="basic",
        mult_channels=(1, 2, 4, 2, 2),
        skip_stride=1,
        use_gamma=False,
        pos_embed_dim=24,
        dic_in_channels=None,
        dic_learn_sigma=None,
        dic_use_gamma=None,
        dic3d_pos_embed_dim=None,
        **kwargs,
    ):
        super().__init__()
        if dic_learn_sigma is not None:
            learn_sigma = dic_learn_sigma
        if dic_use_gamma is not None:
            use_gamma = dic_use_gamma
        if dic3d_pos_embed_dim is not None:
            pos_embed_dim = dic3d_pos_embed_dim

        self.skip_stride = skip_stride
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.input_size = input_size
        self.pos_embed_dim = pos_embed_dim

        self.levels = 3

        if len(depth) == 3:
            depth = depth + [depth[1] + 1, depth[0] + 1]
        elif len(depth) == 4:
            depth = depth + [depth[2] + 1, depth[1] + 1, depth[0] + 1]
            self.levels = 4
        elif len(depth) == 7:
            self.levels = 4
        assert depth[:self.levels] == depth[-self.levels:][::-1]
        if len(mult_channels) == 3:
            mult_channels = mult_channels + mult_channels[:-1][::-1]
        elif len(mult_channels) == 4:
            mult_channels = mult_channels + mult_channels[:-1][::-1]

        self.voxelization = Voxelization(resolution=input_size, normalize=True, eps=0)

        in_chans = in_channels + (pos_embed_dim if pos_embed_dim else 0)
        self.x_embedder = OverlapPatchEmbed3d(in_chans, hidden_size * mult_channels[0], bias=True)

        self.t_embedder_ls = nn.ModuleList([TimestepEmbedder(hidden_size * mult) for mult in mult_channels[:self.levels]])
        self.y_embedder_ls = nn.ModuleList(
            [LabelEmbedder(num_classes, hidden_size * mult, class_dropout_prob) for mult in mult_channels[:self.levels]]
        )

        self.enc_blocks = nn.ModuleList()
        self.lat_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        stages = self.levels - 1

        # encoder
        kwargs = dict(kwargs)
        kwargs["use_gamma"] = use_gamma

        for level_idx, mult, next_mult in zip(range(stages), mult_channels[:stages], mult_channels[1:stages + 1]):
            channel_size = int(hidden_size * mult)
            self.enc_blocks.append(
                nn.ModuleList([
                    U_Block(input_size // (2**level_idx), channel_size, **kwargs) for _ in range(depth[level_idx])
                ])
            )
            self.downs.append(Downsample3d(channel_size, hidden_size * next_mult))

        # latent
        channel_size = int(hidden_size * mult_channels[stages])
        self.lat_blocks.append(
            nn.ModuleList([
                U_Block(input_size // (2**stages), channel_size, **kwargs) for _ in range(depth[stages])
            ])
        )

        # decoder
        for level_idx, mult, skip_mult in zip(list(range(stages))[::-1], mult_channels[-stages:], mult_channels[:stages][::-1]):
            self.ups.append(Upsample3d(channel_size, hidden_size * mult))
            in_channel_size = hidden_size * mult + hidden_size * skip_mult
            channel_size = int(hidden_size * mult)
            self.dec_blocks.append(
                nn.ModuleList([
                    U_Block(
                        input_size // (2**level_idx),
                        channel_size,
                        input_chans=in_channel_size if blk_idx % self.skip_stride == 0 else None,
                        **kwargs,
                    )
                    for blk_idx in range(depth[-level_idx - 1])
                ])
            )

        # last stage condition need special treatment
        self.last_stage_cond_idx = mult_channels[:self.levels].index(mult_channels[-1])

        self.output = nn.Conv3d(channel_size, channel_size, kernel_size=3, stride=1, padding=1, bias=True)

        self.final_layer = FinalLayer3d(channel_size, self.out_channels)
        self.initialize_weights()

        if pos_embed_dim:
            pos = get_3d_sincos_pos_embed(pos_embed_dim, input_size)
            pos = torch.from_numpy(pos).float()
            pos = pos.view(input_size, input_size, input_size, pos_embed_dim).permute(3, 0, 1, 2).unsqueeze(0)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            self.pos_embed = None

        # check
        assert len(mult_channels) == len(depth) == self.levels * 2 - 1

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        for y_embedder in self.y_embedder_ls:
            nn.init.normal_(y_embedder.embedding_table.weight, std=0.02)

        for t_embedder in self.t_embedder_ls:
            nn.init.normal_(t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def _add_positional_channels(self, x):
        if self.pos_embed is None:
            return x
        pos = self.pos_embed.to(dtype=x.dtype, device=x.device)
        return torch.cat([x, pos.expand(x.shape[0], -1, -1, -1, -1)], dim=1)

    def _inflate_conv_weight(self, w2d, target_shape):
        out_ch, in_ch, k_d, k_h, k_w = target_shape
        if w2d.shape[0] != out_ch or w2d.shape[2] != k_h or w2d.shape[3] != k_w:
            return None

        base_in = min(in_ch, w2d.shape[1])
        w3d = torch.zeros(target_shape, dtype=w2d.dtype)
        w3d[:, :base_in] = w2d[:, :base_in].unsqueeze(2).repeat(1, 1, k_d, 1, 1) / k_d
        return w3d

    def inflate_state_dict_from_2d(self, state_2d):
        model_state = self.state_dict()
        inflated = {}
        skipped = []
        for key, value in state_2d.items():
            if key not in model_state:
                skipped.append(key)
                continue
            target = model_state[key]
            if target.shape == value.shape:
                inflated[key] = value
                continue
            if value.ndim == 4 and target.ndim == 5:
                w3d = self._inflate_conv_weight(value, target.shape)
                if w3d is not None:
                    inflated[key] = w3d
                    continue
            skipped.append(key)
        return inflated, skipped

    def load_dic_checkpoint(self, path, use_ema=False, filter_mismatch=True):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            if use_ema and "ema" in ckpt:
                state = ckpt["ema"]
            elif "model" in ckpt:
                state = ckpt["model"]
        if filter_mismatch:
            state, skipped = self.inflate_state_dict_from_2d(state)
        else:
            skipped = []
        msg = self.load_state_dict(state, strict=False)
        return msg, skipped

    def forward(self, x, t, y):
        features, coords = x, x
        x, voxel_coords = self.voxelization(features, coords)
        x = self._add_positional_channels(x)
        x = self.x_embedder(x)

        cond_ls = []
        for idx in range(self.levels):
            t_emb = self.t_embedder_ls[idx](t)
            if idx == 0:
                y_emb, y_dropped = self.y_embedder_ls[idx](y, self.training)
            else:
                y_emb, _ = self.y_embedder_ls[idx](y_dropped, False)
            cond_ls.append(t_emb + y_emb)

        c_ls = cond_ls + cond_ls[1:-1][::-1] + [cond_ls[self.last_stage_cond_idx]]

        skip = []
        stage_idx = 0

        for idx, stage in enumerate(self.enc_blocks):
            for blk_idx, block in enumerate(stage):
                x = block(x, c_ls[stage_idx])
                if (len(stage) - 1 - blk_idx) % self.skip_stride == 0:
                    skip.append(x)

            stage_idx += 1
            x = self.downs[idx](x)

        for stage in self.lat_blocks:
            for block in stage:
                x = block(x, c_ls[stage_idx])
            stage_idx += 1

        for idx, stage in enumerate(self.dec_blocks):
            x = self.ups[idx](x)
            for blk_idx, block in enumerate(stage):
                if blk_idx % self.skip_stride == 0:
                    x = block(torch.cat([x, skip.pop()], 1), c_ls[stage_idx])
                else:
                    x = block(x, c_ls[stage_idx])
            stage_idx += 1

        x = self.output(x)
        x = self.final_layer(x, c_ls[stage_idx - 1])
        x = voxel_ops.trilinear_devoxelize(x, voxel_coords, self.input_size, self.training)
        return x


def _dic3dconv_builder(model_name):
    def _build(**kwargs):
        return DiC3DConv(**kwargs)
    return _build


def DiC3DConv_S(**kwargs):
    return DiC3DConv(depth=[6, 6, 5, 6, 6], hidden_size=96, **kwargs)


def DiC3DConv_B(**kwargs):
    return DiC3DConv(depth=[6, 6, 5, 6, 6], hidden_size=192, **kwargs)


def DiC3DConv_XL(**kwargs):
    return DiC3DConv(depth=[7, 7, 8, 7, 7], hidden_size=384, **kwargs)


def DiC3DConv_H(**kwargs):
    return DiC3DConv(depth=[14, 14, 10, 14, 14], hidden_size=384, **kwargs)


DiC3DConv_models = {
    "DiC3DConv-S": DiC3DConv_S,
    "DiC3DConv-B": DiC3DConv_B,
    "DiC3DConv-XL": DiC3DConv_XL,
    "DiC3DConv-H": DiC3DConv_H,
}

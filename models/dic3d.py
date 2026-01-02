import os
import sys

import torch
import torch.nn as nn

from modules.voxelization import Voxelization
import modules.functional as F
from models.dic3d_conv import DiC3DConv_models


def _ensure_dic_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dic_root = os.path.join(repo_root, "DiC")
    if dic_root not in sys.path:
        sys.path.insert(0, dic_root)


try:
    from dic_models import DiC_models
except ModuleNotFoundError:
    _ensure_dic_on_path()
    from dic_models import DiC_models


class DiC3D(nn.Module):
    def __init__(
        self,
        model_name,
        input_size=32,
        in_channels=3,
        num_classes=1,
        dic_in_channels=None,
        dic_learn_sigma=False,
        dic_use_gamma=False,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.dic_in_channels = dic_in_channels if dic_in_channels is not None else in_channels
        self.dic_learn_sigma = dic_learn_sigma

        self.voxelization = Voxelization(resolution=input_size, normalize=True, eps=0)
        kwargs.pop("dic3d_pos_embed_dim", None)
        self.model = DiC_models[model_name](
            input_size=input_size,
            in_channels=self.dic_in_channels,
            num_classes=num_classes,
            learn_sigma=dic_learn_sigma,
            use_gamma=dic_use_gamma,
            **kwargs
        )

        if self.dic_in_channels != in_channels:
            self.input_adapter = nn.Conv2d(in_channels, self.dic_in_channels, kernel_size=1, bias=True)
        else:
            self.input_adapter = nn.Identity()

        dic_out_channels = self.dic_in_channels * (2 if dic_learn_sigma else 1)
        if dic_out_channels != in_channels:
            self.output_adapter = nn.Conv2d(dic_out_channels, in_channels, kernel_size=1, bias=True)
        else:
            self.output_adapter = nn.Identity()

    def _voxel_to_slices(self, x):
        # (B, C, X, Y, Z) -> (B*Z, C, X, Y)
        b, c, x_size, y_size, z_size = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(b * z_size, c, x_size, y_size)
        return x, z_size

    def _slices_to_voxel(self, x, b, z_size):
        # (B*Z, C, X, Y) -> (B, C, X, Y, Z)
        c, x_size, y_size = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, z_size, c, x_size, y_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x

    def forward(self, x, t, y):
        features, coords = x, x
        voxels, voxel_coords = self.voxelization(features, coords)

        b = voxels.shape[0]
        slices, z_size = self._voxel_to_slices(voxels.float())
        slices = self.input_adapter(slices)

        t_rep = t.repeat_interleave(z_size)
        y_rep = y.repeat_interleave(z_size)
        out = self.model(slices, t_rep, y_rep)
        out = self.output_adapter(out)

        out = self._slices_to_voxel(out, b, z_size)
        out = F.trilinear_devoxelize(out, voxel_coords, self.input_size, self.training)
        return out

    def load_dic_checkpoint(self, path, use_ema=False, filter_mismatch=True):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            if use_ema and "ema" in ckpt:
                state = ckpt["ema"]
            elif "model" in ckpt:
                state = ckpt["model"]
        if filter_mismatch:
            model_state = self.model.state_dict()
            filtered = {}
            skipped = []
            for key, value in state.items():
                if key not in model_state:
                    skipped.append(key)
                    continue
                if model_state[key].shape != value.shape:
                    skipped.append(key)
                    continue
                filtered[key] = value
            state = filtered
        else:
            skipped = []
        msg = self.model.load_state_dict(state, strict=False)
        return msg, skipped


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    grid = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid, grid, grid)
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape(3, 1, grid_size, grid_size, grid_size)
    return get_3d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    return torch.cat([emb_x, emb_y, emb_z], dim=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


class Lite3DMixer(nn.Module):
    def __init__(self, channels, depth=1, kernel_size=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        pad = kernel_size // 2
        for _ in range(max(depth, 0)):
            self.blocks.append(nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=pad, groups=channels, bias=False),
                nn.SiLU(),
                nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            ))

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class DiC3DMix(nn.Module):
    def __init__(
        self,
        model_name,
        input_size=32,
        in_channels=3,
        num_classes=1,
        dic_in_channels=None,
        dic_learn_sigma=False,
        dic_use_gamma=False,
        dic3d_pos_embed_dim=0,
        mix_depth=1,
        mix_kernel_size=3,
        mix_pre=True,
        mix_post=True,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.dic_in_channels = dic_in_channels if dic_in_channels is not None else in_channels
        self.dic_learn_sigma = dic_learn_sigma

        pos_embed_dim = dic3d_pos_embed_dim or 0
        kwargs.pop("dic3d_pos_embed_dim", None)

        self.voxelization = Voxelization(resolution=input_size, normalize=True, eps=0)
        self.model = DiC_models[model_name](
            input_size=input_size,
            in_channels=self.dic_in_channels,
            num_classes=num_classes,
            learn_sigma=dic_learn_sigma,
            use_gamma=dic_use_gamma,
            **kwargs
        )

        if self.dic_in_channels != in_channels:
            self.input_adapter = nn.Conv2d(in_channels, self.dic_in_channels, kernel_size=1, bias=True)
        else:
            self.input_adapter = nn.Identity()

        dic_out_channels = self.dic_in_channels * (2 if dic_learn_sigma else 1)
        if dic_out_channels != in_channels:
            self.output_adapter = nn.Conv2d(dic_out_channels, in_channels, kernel_size=1, bias=True)
        else:
            self.output_adapter = nn.Identity()

        if pos_embed_dim > 0:
            pos = get_3d_sincos_pos_embed(pos_embed_dim, input_size)
            pos = pos.view(input_size, input_size, input_size, pos_embed_dim).permute(3, 0, 1, 2).unsqueeze(0)
            self.register_buffer("pos_embed", pos, persistent=False)
            self.pos_proj = nn.Conv3d(in_channels + pos_embed_dim, in_channels, kernel_size=1, bias=True)
        else:
            self.pos_embed = None
            self.pos_proj = nn.Identity()

        self.pre_mix = Lite3DMixer(in_channels, depth=mix_depth, kernel_size=mix_kernel_size) if mix_pre else nn.Identity()
        self.post_mix = Lite3DMixer(in_channels, depth=mix_depth, kernel_size=mix_kernel_size) if mix_post else nn.Identity()

    def _voxel_to_slices(self, x):
        # (B, C, X, Y, Z) -> (B*Z, C, X, Y)
        b, c, x_size, y_size, z_size = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(b * z_size, c, x_size, y_size)
        return x, z_size

    def _slices_to_voxel(self, x, b, z_size):
        # (B*Z, C, X, Y) -> (B, C, X, Y, Z)
        c, x_size, y_size = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, z_size, c, x_size, y_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x

    def _add_positional_channels(self, x):
        if self.pos_embed is None:
            return x
        pos = self.pos_embed.to(dtype=x.dtype, device=x.device)
        x = torch.cat([x, pos.expand(x.shape[0], -1, -1, -1, -1)], dim=1)
        return self.pos_proj(x)

    def forward(self, x, t, y):
        features, coords = x, x
        voxels, voxel_coords = self.voxelization(features, coords)
        voxels = voxels.float()
        voxels = self._add_positional_channels(voxels)
        voxels = self.pre_mix(voxels)

        b = voxels.shape[0]
        slices, z_size = self._voxel_to_slices(voxels)
        slices = self.input_adapter(slices)

        t_rep = t.repeat_interleave(z_size)
        y_rep = y.repeat_interleave(z_size)
        out = self.model(slices, t_rep, y_rep)
        out = self.output_adapter(out)

        out = self._slices_to_voxel(out, b, z_size)
        out = self.post_mix(out)
        out = F.trilinear_devoxelize(out, voxel_coords, self.input_size, self.training)
        return out

    def load_dic_checkpoint(self, path, use_ema=False, filter_mismatch=True):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            if use_ema and "ema" in ckpt:
                state = ckpt["ema"]
            elif "model" in ckpt:
                state = ckpt["model"]
        if filter_mismatch:
            model_state = self.model.state_dict()
            filtered = {}
            skipped = []
            for key, value in state.items():
                if key not in model_state:
                    skipped.append(key)
                    continue
                if model_state[key].shape != value.shape:
                    skipped.append(key)
                    continue
                filtered[key] = value
            state = filtered
        else:
            skipped = []
        msg = self.model.load_state_dict(state, strict=False)
        return msg, skipped


def _dic3d_builder(model_name):
    def _build(**kwargs):
        return DiC3D(model_name=model_name, **kwargs)
    return _build


def _dic3d_mix_builder(model_name):
    def _build(**kwargs):
        return DiC3DMix(model_name=model_name, **kwargs)
    return _build


DiC3D_models = {name: _dic3d_builder(name) for name in DiC_models}
DiC3D_models.update({
    (name.replace("DiC-", "DiC3DMix-") if name.startswith("DiC-") else f"DiC3DMix-{name}"): _dic3d_mix_builder(name)
    for name in DiC_models
})
DiC3D_models.update(DiC3DConv_models)

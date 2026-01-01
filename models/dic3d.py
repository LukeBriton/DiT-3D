import os
import sys

import torch
import torch.nn as nn

from modules.voxelization import Voxelization
import modules.functional as F


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


def _dic3d_builder(model_name):
    def _build(**kwargs):
        return DiC3D(model_name=model_name, **kwargs)
    return _build


DiC3D_models = {name: _dic3d_builder(name) for name in DiC_models}

import torch
import torch.nn as nn
from torch.nn.functional import interpolate, pad, grid_sample, affine_grid, avg_pool2d, avg_pool1d
import numpy as np
from scattering_transform.wavelet_functions import create_bank, morlet_wavelet, skew_wavelet


class FilterBank(nn.Module):
    def __init__(self, size, num_scales, num_angles):
        super(FilterBank, self).__init__()
        self.size = size
        self.num_scales = num_scales
        self.num_angles = num_angles


class FixedFilterBank(FilterBank):
    def __init__(self, filter_tensor: torch.Tensor):
        assert len(filter_tensor.shape) == 4
        self.filter_tensor = filter_tensor
        num_scales, num_angles, size, _ = self.filter_tensor.shape
        super(FixedFilterBank, self).__init__(size, num_scales, num_angles)

    def to(self, device):
        super(FilterBank, self).to(device)
        self.filter_tensor = self.filter_tensor.to(device)


class Morlet(FixedFilterBank):
    def __init__(self, size, num_scales, num_angles):
        filter_tensor = create_bank(size, num_scales, num_angles, morlet_wavelet)
        super(Morlet, self).__init__(filter_tensor)


class Skew(FixedFilterBank):
    def __init__(self, size, num_scales, num_angles):
        filter_tensor = create_bank(size, num_scales, num_angles, skew_wavelet)
        super(Skew, self).__init__(filter_tensor)


class SubNet(nn.Module):
    def __init__(self, num_ins=2, num_outs=1, hidden_sizes=(16, 16), activation=nn.LeakyReLU):
        super(SubNet, self).__init__()
        layers = []
        sizes = [num_ins] + list(hidden_sizes) + [num_outs]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # input batch, size, size, dim
        b, s, _, dim = x.shape
        x = x.flatten(0, 2)
        x = self.network(x)
        return x.reshape(b, s, s)


class FourierSubNetFilters(FilterBank):

    def __init__(self, size, num_scales, num_angles, subnet=None, scale_invariant=False, init_morlet=False):
        super(FourierSubNetFilters, self).__init__(size, num_scales, num_angles)
        if subnet is None:
            if scale_invariant:
                self.subnet = SubNet()
            else:
                self.subnet = SubNet(num_ins=3)
        else:
            self.subnet = subnet
        self.scale_invariant = scale_invariant

        if num_angles % 2 != 0:
            raise ValueError("num_angles must be even. This allows a significant speedup.")

        self.net_ins = []
        self.scaled_sizes = []
        for scale in range(num_scales):
            scaled_size = self.scale2size(scale)
            self.scaled_sizes.append(scaled_size)

        self.rotation_grids = self._make_grids()
        for scale in range(num_scales):
            grid = self.rotation_grids[scale]
            self.net_ins.append(grid)

        self.update_filters()

        if init_morlet:
            morlet = Morlet(size, num_scales + 1, num_angles)
            self.initialise_weights(morlet.filter_tensor[1:, 5])

    def _make_scaled_filter(self, scale):
        grid = self.net_ins[scale]
        grid = torch.stack([grid[..., 0], grid[..., 1].abs()], dim=-1)
        if not self.scale_invariant:
            grid = torch.cat([grid, scale * torch.ones_like(grid[..., :1])], dim=-1)
        x = self.subnet(grid)
        return torch.cat([x, torch.rot90(x, k=-1, dims=[1, 2])], dim=0)  # rotating 90 saves calcs

    def scale2size(self, scale: float) -> int:
        result = int(self.size / 2 ** scale)
        if result % 2 != 0:
            result += 1
        return result

    def _pad_filters(self, x, scale):
        pad_factor = (self.size - self.scaled_sizes[scale]) // 2
        padded = pad(x, (pad_factor+1, pad_factor, pad_factor+1, pad_factor))  # +1 for the nyq
        return padded

    def update_filters(self):
        filters = []
        for scale in range(self.num_scales):
            scaled_filters = self._make_scaled_filter(scale)
            padded_filters = self._pad_filters(scaled_filters, scale)
            filters.append(padded_filters)
        self.filter_tensor = torch.fft.fftshift(torch.stack(filters), dim=(-2, -1))

    def to(self, device):
        # super(FourierSubNetFilters, self).to(device)
        self.filter_tensor = self.filter_tensor.to(device)
        self.subnet.to(device)
        for j in range(self.num_scales):
            self.net_ins[j] = self.net_ins[j].to(device)
            self.rotation_grids[j] = self.rotation_grids[j].to(device)

    def _make_grids(self) -> list[torch.Tensor]:
        rotation_matrices = []
        for angle in range(self.num_angles // 2):
            theta = angle * np.pi / self.num_angles
            rot_mat = torch.tensor([[np.cos(theta), np.sin(theta), 0],
                                    [-np.sin(theta), np.cos(theta), 0]], dtype=torch.float)
            rotation_matrices.append(rot_mat)
        rotation_matrices = torch.stack(rotation_matrices)

        affine_grids = []
        for scale in range(self.num_scales):
            affine_grids.append(
                affine_grid(
                    rotation_matrices,
                    [self.num_angles // 2, 1, self.scaled_sizes[scale] - 1, self.scaled_sizes[scale] - 1],
                    align_corners=True)
            )

        return affine_grids

    def initialise_weights(self, target, num_epochs=1000):
        optimiser = torch.optim.Adam(self.subnet.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            loss = torch.nn.functional.mse_loss(self.filter_tensor[:, 0], target)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            self.update_filters()



class FourierDirectFilters(FilterBank):
    def __init__(self, size, num_scales, num_angles, init_morlet=False):
        super(FourierDirectFilters, self).__init__(size, num_scales, num_angles)

        if init_morlet:
            morlet = Morlet(size, num_scales + 1, num_angles)
            morlet_filters = morlet.filter_tensor[1:, 2]


        self.scaled_sizes = []
        raw_filters = []
        for scale in range(self.num_scales):
            scaled_size = self.scale2size(scale)
            self.scaled_sizes.append(scaled_size)

            if init_morlet:
                raw = morlet_filters[scale][:scaled_size - 1, :scaled_size // 2].flip(1)
                raw = torch.fft.fftshift(raw, dim=0)
            else:
                raw = torch.nn.Parameter(torch.randn(scaled_size - 1, scaled_size // 2))

            raw_filters.append(raw)

        self.raw_filters = nn.ParameterList(raw_filters)

        if num_angles % 2 != 0:
            raise ValueError("num_angles must be even. This allows a significant speedup.")

        self.filter_tensor = None
        self.rotation_grids = self._make_grids()
        self.update_filters()

    def _make_grids(self) -> list[torch.Tensor]:
        rotation_matrices = []
        for angle in range(self.num_angles // 2):
            theta = angle * np.pi / self.num_angles
            rot_mat = torch.tensor([[np.cos(theta), np.sin(theta), 0],
                                    [-np.sin(theta), np.cos(theta), 0]], dtype=torch.float)
            rotation_matrices.append(rot_mat)
        rotation_matrices = torch.stack(rotation_matrices)

        affine_grids = []
        for scale in range(self.num_scales):
            affine_grids.append(
                affine_grid(
                    rotation_matrices,
                    [self.num_angles // 2, 1, self.scaled_sizes[scale] - 1, self.scaled_sizes[scale] - 1],
                    align_corners=True)
            )

        return affine_grids

    def scale2size(self, scale: float) -> int:
        result = int(self.size / 2 ** scale)
        if result % 2 != 0:
            result += 1
        return result

    def _make_filter(self, scale):
        x = self.raw_filters[scale]
        return torch.cat([x, x.flip(dims=[-1])[:, 1:]], dim=-1)

    def _rotate_filter(self, x, scale):
        grid = self.rotation_grids[scale]
        x = grid_sample(x[None, None, :, :].repeat(self.num_angles // 2, 1, self.scaled_sizes[scale] - 1,
                                                   self.scaled_sizes[scale] - 1), grid).squeeze(1)
        return torch.cat([x, torch.rot90(x, k=-1, dims=[1, 2])], dim=0)  # rotating 90 saves calcs

    def _pad_filters(self, x, scale):
        pad_factor = (self.size - self.scaled_sizes[scale]) // 2
        padded = pad(x, (pad_factor + 1, pad_factor, pad_factor + 1, pad_factor))  # +1 for the nyq
        return padded

    def update_filters(self):
        filters = []
        for scale in range(self.num_scales):
            base_filter = self._make_filter(scale)
            rotated_filters = self._rotate_filter(base_filter, scale)
            padded_filters = self._pad_filters(rotated_filters, scale)
            filters.append(padded_filters)
        self.filter_tensor = torch.fft.fftshift(torch.stack(filters), dim=(-2, -1))

    def to(self, device):
        self.raw_filters = nn.ParameterList(self.raw_filters.to(device))
        self.filter_tensor = self.filter_tensor.to(device)
        for j in range(self.num_scales):
            self.rotation_grids[j] = self.rotation_grids[j].to(device)

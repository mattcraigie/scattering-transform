import torch
import torch.nn as nn
from torch.nn.functional import interpolate, pad, grid_sample, affine_grid, avg_pool2d, avg_pool1d
import numpy as np
from .wavelet_functions import create_bank, morlet_wavelet, skew_wavelet
from .general_functions import clip_fourier_field, dyadic_clip_sizes, morlet_clip_sizes


class FilterBank(nn.Module):
    def __init__(self, size: int, num_scales: int, num_angles: int, clip_sizes: list = None):
        super(FilterBank, self).__init__()
        self.size = size
        self.num_scales = num_scales
        self.num_angles = num_angles
        self.device = None
        self.clip_sizes = clip_sizes
        self.filters = None

    def clip_filters(self):
        assert getattr(self, "filter_tensor", None) is not None, "Must specify filter tensor before calling " \
                                                                 "clip_filters method"
        self.filters = [clip_fourier_field(self.filter_tensor[j], self.clip_sizes[j]) for j in range(self.num_scales)]

    def to(self, device):
        super(FilterBank, self).to(device)
        self.device = device


class FixedFilterBank(FilterBank):
    def __init__(self, filter_tensor: torch.Tensor, clip_sizes: list = None):
        dims = len(filter_tensor.shape)
        assert dims == 4 or dims == 5
        self.filter_tensor = filter_tensor

        num_scales, num_angles, =  filter_tensor.shape[0],  filter_tensor.shape[1]
        size = filter_tensor.shape[-1]

        if clip_sizes is None:
            clip_sizes = [dyadic_clip_sizes(j, size) for j in range(num_scales)]
        else:
            assert len(clip_sizes) == num_scales, "clip_sizes must be the same length as the number of scales"

        super(FixedFilterBank, self).__init__(size, num_scales, num_angles, clip_sizes)

    def to(self, device):
        super(FixedFilterBank, self).to(device)
        self.filter_tensor = self.filter_tensor.to(device)


class Morlet(FixedFilterBank):
    def __init__(self, size, num_scales, num_angles):
        filter_tensor = create_bank(size, num_scales, num_angles, morlet_wavelet)
        super(Morlet, self).__init__(filter_tensor, clip_sizes=[morlet_clip_sizes(j, size) for j in range(num_scales)])
        self.clip_filters()


class ClippedMorlet(Morlet):
    def __init__(self, size, num_scales, num_angles):
        super(ClippedMorlet, self).__init__(size, num_scales, num_angles)

        self.clip_sizes = []
        for j in range(num_scales):
            cs = size // 2 ** j
            self.clip_sizes.append(cs)
            if cs == size:
                continue

            mid = size // 2
            half_cs = cs // 2

            sl = slice(mid - half_cs, mid + half_cs)

            full = torch.fft.fftshift(self.filter_tensor[j])
            new = full[:, sl, sl]
            new_mid = half_cs

            # it is quicker to treat each case separately than work out an algorithm to do this automatically
            # and computationally the same. Ordered around the square left->right for top, then middle, then bottom
            new[:, new_mid:, new_mid:] += full[:, mid-cs:mid-half_cs, mid-cs:mid-half_cs]
            new[:, new_mid:, :] += full[:, mid-cs:mid-half_cs, mid-half_cs:mid+half_cs]
            new[:, new_mid:, :new_mid] += full[:, mid-cs:mid-half_cs, mid+half_cs:mid+cs]
            new[:, :, new_mid:] += full[:, mid-half_cs:mid+half_cs, mid-cs:mid-half_cs]
            new[:, :, :new_mid] += full[:, mid-half_cs:mid+half_cs, mid+half_cs:mid+cs]
            new[:, :new_mid, new_mid:] += full[:, mid+half_cs:mid+cs, mid-cs:mid-half_cs]
            new[:, :new_mid, :] += full[:, mid+half_cs:mid+cs, mid-half_cs:mid+half_cs]
            new[:, :new_mid, :new_mid] += full[:, mid+half_cs:mid+cs, mid+half_cs:mid+cs]

            pad_factor = (size - cs) // 2
            padded = pad(new, (pad_factor, pad_factor, pad_factor, pad_factor))
            self.filter_tensor[j] = torch.fft.fftshift(padded)


class Skew(FixedFilterBank):
    def __init__(self, size, num_scales, num_angles):
        filter_tensor = create_bank(size, num_scales, num_angles, skew_wavelet)
        super(Skew, self).__init__(filter_tensor)


class Box(FixedFilterBank):
    def __init__(self, size, num_scales, num_angles):
        filter_tensor = torch.ones(num_scales, num_angles, size, size)
        super(Box, self).__init__(filter_tensor)


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


def scale2size(full_size, scale):
    result = int(full_size / 2 ** scale)
    if result % 2 != 0:
        result += 1
    return result


def make_grids(size, num_scales, num_angles, clip_sizes):
    affine_grids = []
    for scale in range(num_scales):

        scaled_size = clip_sizes[scale]

        rotation_matrices = []
        for angle in range(num_angles // 2):
            theta = angle * np.pi / num_angles
            rot_mat = torch.tensor([[np.cos(theta), np.sin(theta), 0],
                                    [-np.sin(theta), np.cos(theta), 0]], dtype=torch.float)
            rotation_matrices.append(rot_mat)
        rotation_matrices = torch.stack(rotation_matrices)

        grids = torch.nn.functional.affine_grid(rotation_matrices,
                                                [num_angles // 2, 1, scaled_size + 1, scaled_size + 1],
                                                align_corners=True)
        affine_grids.append(grids)

    return affine_grids


def pad_filters(x, full_size, scaled_size):
    if full_size == scaled_size - 1:
        return x
    pad_factor = (full_size - scaled_size) // 2
    padded = pad(x, (pad_factor, pad_factor, pad_factor, pad_factor))  # +1 for the nyq
    return padded


def make_duplicate_rotations(x):
    return torch.cat([x, torch.rot90(x, k=-1, dims=[1, 2])], dim=0)  # rotating 90 saves calcs


def make_duplicate_rotations_full(x):
    dr = make_duplicate_rotations(x)
    return torch.cat([dr, torch.rot90(dr, k=-2, dims=[1, 2])], dim=0)  # rotating 180 saves calcs


def crop_extra_nyquist(x):
    # the rotations work by rotating about the centre then cropping out the bottom and right 'extra' freqs
    # this retains the nyquist (which is in the topmost row and leftmost column) and allows us to use
    # the rot90 above to save calcs.
    return x[:, :-1, :-1]


def make_filters(grids, num_scales, full_size, filter_func, clip_sizes, full_rotation=False):
    # full rotation makes it so that the filters are rotated all the way around, not just 180 degrees
    filters = []
    for scale in range(num_scales):
        half_rotated_filters = filter_func(grids, scale)

        if full_rotation:
            fully_rotated_filters = make_duplicate_rotations_full(half_rotated_filters)
        else:
            fully_rotated_filters = make_duplicate_rotations(half_rotated_filters)

        nyquist_cropped_filters = crop_extra_nyquist(fully_rotated_filters)
        padded_filters = pad_filters(nyquist_cropped_filters, full_size, clip_sizes[scale])
        filters.append(padded_filters)
    return torch.fft.fftshift(torch.stack(filters), dim=(-2, -1))


class GridFuncFilter(FilterBank):
    def __init__(self, size, num_scales, num_angles, clip_sizes=None, full_rotation=False):
        super(GridFuncFilter, self).__init__(size, num_scales, num_angles)

        self.full_rotation = full_rotation

        if num_angles % 2 != 0:
            raise ValueError("num_angles must be even. This allows a significant speedup.")

        if clip_sizes is None:
            self.clip_sizes = [dyadic_clip_sizes(scale, size) for scale in range(num_scales)]
        else:
            self.clip_sizes = clip_sizes
        self.grids = make_grids(size, num_scales, num_angles, self.clip_sizes)
        self.filter_tensor = None

    def update_filters(self):
        self.filter_tensor = make_filters(self.grids, self.num_scales, self.size, self.filter_function,
                                          self.clip_sizes, full_rotation=self.full_rotation)
        self.clip_filters()

    def filter_function(self, grid, scale):
        # takes in grid and scale and returns a filter
        # should be overwritten in subclasses
        return torch.ones_like(grid)

    def to(self, device):
        super(GridFuncFilter, self).to(device)
        self.grids = [g.to(device) for g in self.grids]
        self.filter_tensor = self.filter_tensor.to(device)


class BandPass(GridFuncFilter):
    def __init__(self, size, num_scales, num_angles):
        super(BandPass, self).__init__(size, num_scales, num_angles)
        self.update_filters()

    def filter_function(self, grid, scale):
        radius = torch.sqrt(grid[scale][:, :, :, 0] ** 2 + grid[scale][:, :, :, 1] ** 2)
        angle = torch.atan2(grid[scale][:, :, :, 1], grid[scale][:, :, :, 0])
        mask = (radius > 0.5) & (radius < 1) & (angle > -np.pi / 8) & (angle < np.pi / 8)
        filters = torch.zeros_like(grid[scale])[:, :, :, 0]
        filters[mask] = 1
        return filters


class LowPass(GridFuncFilter):
    def __init__(self, size, num_scales, num_angles):
        super(LowPass, self).__init__(size, num_scales, num_angles)
        self.update_filters()

    def filter_function(self, grid, scale):
        radius = torch.sqrt(grid[scale][:, :, :, 0] ** 2 + grid[scale][:, :, :, 1] ** 2)
        angle = torch.atan2(grid[scale][:, :, :, 1], grid[scale][:, :, :, 0])
        mask = (radius < 1) & (angle > -np.pi / 8) & (angle < np.pi / 8)
        filters = torch.zeros_like(grid[scale])[:, :, :, 0]
        filters[mask] = 1
        return filters


class FourierSubNetFilters(GridFuncFilter):

    def __init__(self, size, num_scales, num_angles, subnet=None, scale_invariant=False, init_morlet=True,
                 symmetric=True, periodic=False, clip_sizes=None, full_rotation=False):
        super(FourierSubNetFilters, self).__init__(size, num_scales, num_angles, clip_sizes=clip_sizes, full_rotation=full_rotation)

        if subnet is None:
            if scale_invariant:
                self.subnet = SubNet()
            else:
                self.subnet = SubNet(num_ins=3)
        else:
            self.subnet = subnet
        self.scale_invariant = scale_invariant
        self.symmetric = symmetric
        self.periodic = periodic

        self.net_ins = []
        self.update_filters()

        if init_morlet:
            clipped_morlet = ClippedMorlet(size, num_scales, num_angles)
            self.initialise_weights(clipped_morlet.filter_tensor[:, num_angles - 1])

    def filter_function(self, grid, scale):
        g = grid[scale]

        xpart = g[..., 0]
        ypart = g[..., 1]

        if self.symmetric:
            # ypart = torch.cos(np.pi*g[..., 1])
            ypart = g[..., 1].abs()
        if self.periodic:
            xpart = torch.sin(np.pi * g[..., 0])

        g = torch.stack([xpart, ypart], dim=-1)
        if not self.scale_invariant:
            g = torch.cat([g, scale * torch.ones_like(g[..., :1])], dim=-1)
        filters = self.subnet(g)
        return filters

    def to(self, device):
        super(FourierSubNetFilters, self).to(device)
        self.subnet.to(device)

    def initialise_weights(self, target, num_epochs=1000):
        optimiser = torch.optim.Adam(self.subnet.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            loss = torch.nn.functional.mse_loss(self.filter_tensor[:, 0], target)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            self.update_filters()




class TrainableMorlet(GridFuncFilter):
    def __init__(self, size, num_scales, num_angles, scale_invariant=True, enforce_symmetry=True):
        super(TrainableMorlet, self).__init__(size, num_scales, num_angles)

        self.scale_invariant = scale_invariant
        self.enforce_symmetry = enforce_symmetry

        if scale_invariant:
            self.a = torch.nn.Parameter(-torch.rand(1) - 1)
            if enforce_symmetry:
                self.b = torch.zeros(1)
            else:
                self.b = torch.nn.Parameter(torch.rand(1) - 0.5)

            self.c = torch.nn.Parameter(-torch.rand(1) - 1)
            self.kr = torch.nn.Parameter(torch.rand(1) * 0.8)
        else:
            self.a = torch.nn.Parameter(-torch.rand(num_scales, 1) - 1)
            if enforce_symmetry:
                self.b = torch.zeros(num_scales, 1)
            else:
                self.b = torch.nn.Parameter(torch.rand(num_scales, 1) - 0.5)
            self.c = torch.nn.Parameter(-torch.rand(num_scales, 1) - 1)
            self.kr = torch.nn.Parameter(torch.rand(num_scales, 1) * 0.8)

        self.update_filters()

    def filter_function(self, grid, scale):
        a = self.a if self.scale_invariant else self.a[scale]
        b = self.b if self.scale_invariant else self.b[scale]
        c = self.c if self.scale_invariant else self.c[scale]
        kr = self.kr if self.scale_invariant else self.kr[scale]

        filters = self.morlet_function(grid[scale], a, b, c, kr)
        return filters

    def morlet_function(self, k_grid, a, b, c, kr):

        k_grid = k_grid.unsqueeze(-2)

        # Using a Cholesky decomposition to ensure that the covariance matrix is positive definite
        # and learn the covariance in a more convex space
        # The L matrix is [[a, 0], [b, c]] where L @ L^T = covariance matrix

        # a and c are strictly positive, b is unconstrained
        a = nn.functional.softplus(a)
        c = nn.functional.softplus(c)

        # I benchmarked and this is actually faster than analytically writing down the inverse in terms of a, b and c
        cholesky_lower_triangle = torch.stack([a, torch.zeros_like(a), b, c], dim=-1).reshape(-1, 2, 2)
        covariance_matrix = torch.matmul(cholesky_lower_triangle, cholesky_lower_triangle.transpose(-1, -2))
        inv_covariance_matrix = torch.inverse(covariance_matrix)

        # k0 vec
        k0_vec = torch.stack([kr, torch.zeros_like(kr)], dim=-1).unsqueeze(0).to(k_grid.device)


        # compute the morlet wavelet on the grid k
        gaussian_at_k0 = torch.exp(-(k_grid - k0_vec) @ inv_covariance_matrix @ ((k_grid - k0_vec).transpose(-1, -2)) / 2)
        gaussian_at_k = torch.exp(-(k_grid @ inv_covariance_matrix @ (k_grid.transpose(-1, -2))) / 2)
        admissibility = torch.exp(-k0_vec @ inv_covariance_matrix @ (k0_vec.transpose(-1, -2)) / 2)
        morlet = gaussian_at_k0 - admissibility * gaussian_at_k

        return morlet.squeeze(-1).squeeze(-1)

    def to(self, device):
        super(TrainableMorlet, self).to(device)
        if self.enforce_symmetry:
            self.b = self.b.to(device)







class FourierDirectFilters(FilterBank):
    def __init__(self, size, num_scales, num_angles, init_morlet=False):
        super(FourierDirectFilters, self).__init__(size, num_scales, num_angles)

        if init_morlet:
            morlet = Morlet(size, num_scales + 1, num_angles)
            morlet_filters = morlet.filter_tensor[1:, num_angles // 2 - 1]


        self.clip_sizes = []
        raw_filters = []
        for scale in range(self.num_scales):
            scaled_size = self.scale2size(scale)
            self.clip_sizes.append(scaled_size)

            if init_morlet:
                raw = morlet_filters[scale][:scaled_size - 1, :scaled_size // 2].flip(1)
                raw = torch.fft.fftshift(raw, dim=0)
            else:
                raw = torch.randn(scaled_size - 1, scaled_size // 2)

            raw_filters.append(torch.nn.Parameter(raw))

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
                    [self.num_angles // 2, 1, self.clip_sizes[scale] - 1, self.clip_sizes[scale] - 1],
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
        x = grid_sample(x[None, None, :, :].repeat(self.num_angles // 2, 1, self.clip_sizes[scale] - 1,
                                                   self.clip_sizes[scale] - 1), grid).squeeze(1)
        return torch.cat([x, torch.rot90(x, k=-1, dims=[1, 2])], dim=0)  # rotating 90 saves calcs

    def _pad_filters(self, x, scale):
        pad_factor = (self.size - self.clip_sizes[scale]) // 2
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


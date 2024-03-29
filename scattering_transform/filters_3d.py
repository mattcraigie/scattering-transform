import torch
import torch.nn as nn
from torch.nn.functional import interpolate, pad, grid_sample, affine_grid, avg_pool2d, avg_pool1d
import numpy as np
from .wavelet_functions import create_bank, morlet_wavelet, skew_wavelet
from .general_functions import clip_fourier_field_3d, dyadic_clip_sizes, morlet_clip_sizes
from .filters import SubNet



# these are the healpy pixel centres for the 12 pixels in the base resolution
# I don't want to need a healpy dependence for this, so I'm just hardcoding them here
# format is (theta, phi) in radians where theta in [0, pi] and phi in [0, 2pi]
pixel_centres_12 = [(0.8410686705679303, 0.7853981633974483),  # 0
                 (0.8410686705679303, 2.356194490192345), # 1
                 (0.8410686705679303, 3.926990816987241), # 2
                 (0.8410686705679303, 5.497787143782138), # 3
                 (1.5707963267948966, 0.0), # 4
                 (1.5707963267948966, 1.5707963267948966), # 5
                 (1.5707963267948966, 3.141592653589793), # 6
                 (1.5707963267948966, 4.71238898038469), # 7
                 (2.300523983021863, 0.7853981633974483), # 8
                 (2.300523983021863, 2.356194490192345), # 9
                 (2.300523983021863, 3.926990816987241), # 10
                 (2.300523983021863, 5.497787143782138)] # 11

# here, I keep points on the top 4 corners and top and front faces
pixel_centres_6 = [(0.8410686705679303, 0.7853981633974483),   # 0
                 (0.8410686705679303, 2.356194490192345),  # 1
                 (1.5707963267948966, 0.0),  # 4
                 (1.5707963267948966, 1.5707963267948966),  # 5
                 (2.300523983021863, 0.7853981633974483),  # 8
                 (2.300523983021863, 2.356194490192345),  # 9
]


class FilterBank3d(nn.Module):
    def __init__(self, size: int, num_scales: int, num_angles: int, clip_sizes: list = None):
        super(FilterBank3d, self).__init__()
        self.size = size
        self.num_scales = num_scales
        self.num_angles = num_angles
        self.device = None
        self.clip_sizes = clip_sizes
        self.filters = None

    def clip_filters(self):
        assert getattr(self, "filter_tensor", None) is not None, "Must specify filter tensor before calling " \
                                                                 "clip_filters method"
        self.filters = [clip_fourier_field_3d(self.filter_tensor[j], self.clip_sizes[j]) for j in range(self.num_scales)]

    def to(self, device):
        super(FilterBank3d, self).to(device)
        self.filters = [filt.to(device) for filt in self.filters]
        self.device = device


class FixedFilterBank3d(FilterBank3d):
    def __init__(self, filter_tensor: torch.Tensor, clip_sizes: list = None):
        dims = len(filter_tensor.shape)
        assert dims == 4 or dims == 5
        self.filter_tensor = filter_tensor

        num_scales, num_angles, = filter_tensor.shape[0],  filter_tensor.shape[1]
        size = filter_tensor.shape[-1]

        if clip_sizes is None:
            clip_sizes = [dyadic_clip_sizes(j, size) for j in range(num_scales)]
        else:
            assert len(clip_sizes) == num_scales, "clip_sizes must be the same length as the number of scales"

        super(FixedFilterBank3d, self).__init__(size, num_scales, num_angles, clip_sizes)

    def to(self, device):
        super(FixedFilterBank3d, self).to(device)
        self.filter_tensor = self.filter_tensor.to(device)


def make_grids_3d(sizes, num_scales, num_angles):
    affine_grids = []

    if num_angles == 12:
        angles = pixel_centres_12
    elif num_angles == 6:
        angles = pixel_centres_6
    else:
        raise ValueError("num_angles must be either 6 or 12")

    for scale in range(num_scales):
        scaled_size = sizes[scale]

        rotation_matrices = []
        for angle in angles:
            theta_x, theta_y = angle
            rot_mat = torch.tensor([
                [np.cos(theta_y), np.sin(theta_x) * np.sin(theta_y), np.cos(theta_x) * np.sin(theta_y), 0],
                [0, np.cos(theta_x), -np.sin(theta_x), 0],
                [-np.sin(theta_y), np.sin(theta_x) * np.cos(theta_y), np.cos(theta_x) * np.cos(theta_y), 0],
            ], dtype=torch.float)
            rotation_matrices.append(rot_mat)
        rotation_matrices = torch.stack(rotation_matrices)

        grids = torch.nn.functional.affine_grid(rotation_matrices, [len(angles), 1, scaled_size + 1, scaled_size + 1, scaled_size + 1], align_corners=True)
        affine_grids.append(grids)

    return affine_grids


def pad_filters(x, full_size, scaled_size):
    if full_size == scaled_size - 1:
        return x
    pad_factor = (full_size - scaled_size) // 2
    padded = pad(x, (pad_factor, pad_factor, pad_factor, pad_factor, pad_factor, pad_factor))  # +1 for the nyq
    return padded


def crop_extra_nyquist(x):
    # I think I need this to enable the absolute value symmetry.
    return x[:, :-1, :-1, :-1]


def make_filters(grids, num_scales, full_size, filter_func, clip_sizes):
    filters = []
    for scale in range(num_scales):
        rotated_filters = filter_func(grids, scale)
        nyquist_cropped_filters = crop_extra_nyquist(rotated_filters)
        padded_filters = pad_filters(nyquist_cropped_filters, full_size, clip_sizes[scale])
        filters.append(padded_filters)
    return torch.fft.fftshift(torch.stack(filters), dim=(-3, -2, -1))


class GridFuncFilter3d(FilterBank3d):
    def __init__(self, size, num_scales, num_angles, clip_sizes=None):
        super(GridFuncFilter3d, self).__init__(size, num_scales, num_angles)

        if clip_sizes is None:
            self.clip_sizes = [dyadic_clip_sizes(scale, size) for scale in range(num_scales)]
        else:
            self.clip_sizes = clip_sizes
        self.grids = make_grids_3d(self.clip_sizes, num_scales, num_angles)
        self.filter_tensor = None

    def update_filters(self):
        self.filter_tensor = make_filters(self.grids, self.num_scales, self.size, self.filter_function,
                                          self.clip_sizes)
        self.clip_filters()

    def filter_function(self, grids, scale):
        g = grids[scale]
        # takes in grid and scale and returns a filter
        # should be overwritten in subclasses
        return torch.ones_like(g[:, :, :, :, 0])

    def to(self, device):
        super(GridFuncFilter3d, self).to(device)
        self.grids = [g.to(device) for g in self.grids]
        self.filter_tensor = self.filter_tensor.to(device)


class SubNet3d(nn.Module):
    def __init__(self, num_ins=2, num_outs=1, hidden_sizes=(16, 16), activation=nn.LeakyReLU):
        super(SubNet3d, self).__init__()
        layers = []
        sizes = [num_ins] + list(hidden_sizes) + [num_outs]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # input batch, size, size, dim
        b, s, _, _, dim = x.shape
        x = x.flatten(0, 2)
        x = self.network(x)
        return x.reshape(b, s, s, s)


class FourierSubNetFilters3d(GridFuncFilter3d):

    def __init__(self, size, num_scales, subnet=None,
                 symmetric=True, clip_sizes=None):
        num_angles = 6 if symmetric else 12

        super(FourierSubNetFilters3d, self).__init__(size, num_scales, num_angles, clip_sizes=clip_sizes)

        if subnet is None:
            self.subnet = SubNet3d(num_ins=4, hidden_sizes=(64, 64))
        else:
            self.subnet = subnet

        self.symmetric = symmetric

        self.net_ins = []
        self.update_filters()

    def filter_function(self, grid, scale):
        g = grid[scale]

        xpart = g[..., 0]
        ypart = g[..., 1]
        zpart = g[..., 2]

        if self.symmetric:
            xpart = xpart.abs()
            ypart = ypart.abs()

        g = torch.stack([xpart, ypart, zpart], dim=-1)
        g = torch.cat([g, scale * torch.ones_like(g[..., :1])], dim=-1)
        filters = self.subnet(g)
        return filters

    def to(self, device):
        super(FourierSubNetFilters3d, self).to(device)
        self.subnet.to(device)
        self.update_filters()

    def initialise_weights(self, target, num_epochs=1000):
        optimiser = torch.optim.Adam(self.subnet.parameters(), lr=0.001)

        print_every = num_epochs // 10
        for epoch in range(num_epochs):

            loss = torch.nn.functional.mse_loss(self.filter_tensor[:, 0], target)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            self.update_filters()

            # use sceintific notation for loss
            if epoch % print_every == 0:
                print(f'Epoch {epoch} \t| Loss {loss.item():.2e}')


class Morlet3d(GridFuncFilter3d):
    def __init__(self, size, num_scales, k0=None, covariance=None):
        super(Morlet3d, self).__init__(size, num_scales, num_angles=6)

        self.k0 = k0
        self.covariance = covariance

        if self.k0 is None:
            self.k0 = torch.tensor([0, 0, 1], dtype=torch.float32).unsqueeze(0) * 0.75
        if self.covariance is None:
            self.covariance = torch.eye(3, dtype=torch.float32) * 20

        self.update_filters()

    def filter_function(self, grids, scale):
        g = grids[scale]
        return self._morlet(g.unsqueeze(-2), self.k0, self.covariance)

    def _gaussian(self, k, covariance):
        return torch.exp(-0.5 * k @ covariance @ k.transpose(-1, -2))  # shape (batch, size, size, size)

    def _gabor(self, k, k0, covariance):
        delta = k - k0[None, None, None, None, :, :]  # shape (batch, size, size, size, 1, 3)
        return self._gaussian(delta, covariance).squeeze(-1).squeeze(-1)  # shape (batch, size, size, size)

    def _morlet(self, k, k0, covariance):
        # k is a field of 3d wave vectors (batch, size, size, size, 1, 3)
        # k0 is the center frequency (1, 3)
        # covariance is the covariance matrix (3, 3)
        gabor = self._gabor(k, k0, covariance)
        beta = self._gaussian(k0, covariance)
        gaussian_part = self._gaussian(k, covariance).squeeze(-1).squeeze(-1)
        morlet = gabor - beta * gaussian_part
        return morlet



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
# pixel_centres = [(0.8410686705679303, 0.7853981633974483),  # 0
#                  (0.8410686705679303, 2.356194490192345), # 1
#                  (0.8410686705679303, 3.926990816987241), # 2
#                  (0.8410686705679303, 5.497787143782138), # 3
#                  (1.5707963267948966, 0.0), # 4
#                  (1.5707963267948966, 1.5707963267948966), # 5
#                  (1.5707963267948966, 3.141592653589793), # 6
#                  (1.5707963267948966, 4.71238898038469), # 7
#                  (2.300523983021863, 0.7853981633974483), # 8
#                  (2.300523983021863, 2.356194490192345), # 9
#                  (2.300523983021863, 3.926990816987241), # 10
#                  (2.300523983021863, 5.497787143782138)] # 11

# here, I keep points on the top 4 corners and top and front faces
pixel_centres = [(0.8410686705679303, 0.7853981633974483),   # 0
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


class Morlet3d(FixedFilterBank3d):
    def __init__(self, size, num_scales, num_angles):
        filter_tensor = create_bank(size, num_scales, num_angles, morlet_wavelet)
        filter_tensor = torch.stack([filter_tensor for _ in range(size)], dim=-1)

        # multiply by a gaussian packet along the last dimension.
        # the gaussian packet is centred at the origin and has a standard deviation of 1/8 the size of the filter
        filter_tensor = torch.fft.fftshift(filter_tensor, dim=(-1, -2, -3))

        # create a grid of points in the last dimension
        x = torch.linspace(-size/2, size/2, size)
        x = x.view(1, 1, 1, 1, -1)
        x = x.repeat(num_scales, num_angles, size, size, 1)

        # create the gaussian packet
        # this needs to be fixed so that the packet modulation is different for each scale, but it gets the point across
        sigma = size/8
        gaussian = torch.exp(-x**2 / (2*sigma**2))

        # multiply the filter tensor by the gaussian packet
        filter_tensor = filter_tensor * gaussian

        # shift back
        filter_tensor = torch.fft.ifftshift(filter_tensor, dim=(-1, -2, -3))

        super(Morlet3d, self).__init__(filter_tensor, clip_sizes=[morlet_clip_sizes(j, size) for j in range(num_scales)])
        self.clip_filters()


def make_grids_3d(sizes, num_scales):
    affine_grids = []

    angles = pixel_centres

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


class GridFuncFilter(FilterBank3d):
    def __init__(self, size, num_scales, num_angles, clip_sizes=None):
        super(GridFuncFilter, self).__init__(size, num_scales, num_angles)

        if clip_sizes is None:
            self.clip_sizes = [dyadic_clip_sizes(scale, size) for scale in range(num_scales)]
        else:
            self.clip_sizes = clip_sizes
        self.grids = make_grids_3d(self.clip_sizes, num_scales)
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
        super(GridFuncFilter, self).to(device)
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


class FourierSubNetFilters3d(GridFuncFilter):

    def __init__(self, size, num_scales, num_angles, subnet=None,
                 symmetric=True, clip_sizes=None):
        super(FourierSubNetFilters3d, self).__init__(size, num_scales, num_angles, clip_sizes=clip_sizes)

        if subnet is None:
            self.subnet = SubNet3d(num_ins=4, hidden_sizes=(64, 64))

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



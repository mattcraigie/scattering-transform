import torch
import torch.nn as nn
from torch.nn.functional import interpolate, pad, grid_sample, affine_grid, avg_pool2d, avg_pool1d
import numpy as np
from wavelet_functions import create_bank, morlet_wavelet


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


class TrainableFilterBank(FilterBank):
    def __init__(self,
                 size: int = None,
                 num_scales: int = None,
                 num_angles: int = None):

        super(TrainableFilterBank, self).__init__(size, num_scales, num_angles)
        self.filter_tensor = None

    def save_filters(self, filter_path: str = 'filter.pt'):
        assert self.filter_tensor is not None, "Cannot save because the filter_tensor is None"
        torch.save(self.filter_tensor, filter_path)


class FourierTrainable(TrainableFilterBank):
    def __init__(self,
                 size: int,
                 num_scales: int,
                 num_angles: int,
                 base_scale: int = 0,
                 init_filter: torch.Tensor = None,
                 down_method: str = 'avg_pool',
                 up_method: str = 'interp'):
        """
        A trainable filter bank in Fourier space, to be used for the scattering transform. The base filter is trainable,
        (and acts as the scattering transform's mother wavelet) and is resized and rotated to the various filter
        scales and angles (often denoted by J and L)

        Parameters
        ----------
        size : int
            The size of the square filter.
        num_scales : int
            The number of scales (filter sizes) to use.
        num_angles : int
            The number of rotations to use.
        base_scale : int
            The scale of the smallest filter, which is assumed to be square.
        init_filter : torch.Tensor
            The initial filter to use, which should have the same dimensions as the smallest filter.
        down_method : str
            The method used to downsample the filter during scaling. Either 'avg_pool' or 'interp'.
        up_method : str
            The method used to upsample the filter during scaling. Only 'interp' is currently supported.

        Returns
        -------
        torch.nn.Module
            A PyTorch module object.
        """

        super(FourierTrainable, self).__init__(size, num_scales, num_angles)  # not a super clean flow

        assert size % 2 == 0, "Filter size must be an even number"

        self.down_method = down_method
        self.up_method = up_method

        self.base_scale = base_scale
        self.base_size = self.scale2size(base_scale)
        half = self.base_size // 2

        if init_filter is None:
            self.main_block = nn.Parameter(torch.randn(self.base_size, half - 1), requires_grad=True)
            self.zero_freq = nn.Parameter(torch.randn(self.base_size, 1), requires_grad=True)
            self.nyquist_freq = torch.zeros(self.base_size, 1, requires_grad=False)
            # nn.Parameter(torch.randn(self.base_size, 1), requires_grad=True)

        # make affine grid for num_rotations here, call it during forward
        self.rotation_grid = self._make_affine_grid()

        # update the filters, i.e. scale and rotate the base filter. This fills the filter tensor attribute.
        self.update_filters()

    def update_filters(self):
        """
        Updates the filter tensor. Can be called inside a forward call to update the filters with each training step.

        Returns
        -------
        None
        """
        filters = []
        for scale in range(self.num_scales):
            scaled_filter = self._scale_filter(scale)
            rotated_filters = self._rotate_filter(scaled_filter)
            filters.append(rotated_filters)
        self.filter_tensor = torch.fft.fftshift(torch.stack(filters), dim=(-2, -1))

    def _scale_filter(self, out_scale: int) -> torch.Tensor:
        """
        Scales the base filter to the desired scale.

        Parameters
        ----------
        out_scale : int
            The desired scale of the filter, and integer less than the num_scales attribute.

        Returns
        -------
        torch.Tensor
            The resampled (up or down) and appropriately padded filter tensor.
        """
        if self.base_scale == out_scale:
            return self._arrange_filter(self.main_block, self.zero_freq, self.nyquist_freq)
        # scaling up
        out_size = self.scale2size(out_scale)
        scale_factor = out_size / self.scale2size(self.base_scale)

        if scale_factor < 1:
            main, zero, nyq = self._downsample(scale_factor)
        else:
            main, zero, nyq = self._upsample(scale_factor)

        arranged = self._arrange_filter(main, zero, nyq)

        # applying padding
        pad_factor = (self.size - out_size) // 2
        padded = pad(arranged, (pad_factor, pad_factor, pad_factor, pad_factor))

        return padded

    def _downsample(self, scale_factor: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Downsamples the filter tensor using the specified down_method attribute.

        Parameters
        ----------
        scale_factor : float
            The factor by which to downsample the filter.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The downsampled main filter tensor, zero frequency tensor, and Nyquist frequency tensor.
        """

        if self.down_method == 'interp':
            main_down = interpolate(self.main_block.unsqueeze(0).unsqueeze(0),
                                    scale_factor=(scale_factor, scale_factor), mode='bicubic')[0, 0]
            zero_down = interpolate(self.zero_freq.unsqueeze(0),
                                    scale_factor=(scale_factor,), mode='linear')[0]
            nyq_down = interpolate(self.nyquist_freq.unsqueeze(0),
                                   scale_factor=(scale_factor,), mode='linear')[0]
        elif self.down_method == 'avg_pool':
            inv_factor = int(1 / scale_factor)
            main_down = avg_pool2d(self.main_block.unsqueeze(0).unsqueeze(0),
                                   kernel_size=inv_factor, stride=inv_factor)[0, 0]
            zero_down = avg_pool1d(self.zero_freq.swapaxes(0, 1).unsqueeze(0),
                                   kernel_size=inv_factor, stride=inv_factor)[0].swapaxes(0, 1)
            nyq_down = avg_pool1d(self.nyquist_freq.swapaxes(0, 1).unsqueeze(0),
                                  kernel_size=inv_factor, stride=inv_factor)[0].swapaxes(0, 1)
        else:
            raise ValueError("Invalid value for down_method argument. Allowed values are 'interp' and 'avg_pool'.")
        return main_down, zero_down, nyq_down

    def _upsample(self, scale_factor: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Upsamples the base filter by a scale factor.

        Parameters
        ----------
        scale_factor : float
           The factor by which to upsample the filter.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
           The upsampled main filter tensor, upsampled zero frequency tensor, and upsampled Nyquist frequency tensor.
        """
        # upsampling is a little more annoying... I think
        return NotImplementedError("For now, we can only go large -> small.")
        # print(self.main_block.shape, self.zero_freq.shape)
        # print(scale_factor)
        # if self.up_method == 'interp':
        #     main_up = interpolate(self.main_block.unsqueeze(0).unsqueeze(0),
        #                           scale_factor=scale_factor, mode='bicubic')[0, 0]
        #     zero_up = interpolate(self.zero_freq.swapaxes(0, 1).unsqueeze(0),
        #                           scale_factor=scale_factor, mode='linear')[0].swapaxes(0, 1)
        #     nyq_up = interpolate(self.nyquist_freq.swapaxes(0, 1).unsqueeze(0),
        #                          scale_factor=scale_factor, mode='linear')[0].swapaxes(0, 1)
        #
        #     print(main_up.shape, zero_up.shape)
        # # elifs
        # else:
        #     raise ValueError("Invalid value for up_method argument. Allowed values are 'interp'.")
        # return main_up, zero_up, nyq_up

    def _rotate_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates the filter tensor using the pre-calculated rotation grid.

        Parameters
        ----------
        x : torch.Tensor
            The filter tensor to rotate.

        Returns
        -------
        torch.Tensor
            The rotated filter tensor.
        """
        pass
        # todo: recognise no rotation and rot90 opportunities to speed up
        filters = x[None, None, :, :].expand(self.num_angles, -1, -1, -1)
        return grid_sample(filters, grid=self.rotation_grid).squeeze(1)

    def _arrange_filter(self, main_block: torch.Tensor,
                        zero_freq: torch.Tensor,
                        nyquist_freq: torch.Tensor) -> torch.Tensor:
        res = torch.cat([nyquist_freq,
                         torch.flip(main_block, dims=(1,)),
                         zero_freq,
                         main_block],
                        dim=1)

        return res

    def _make_affine_grid(self) -> torch.Tensor:
        rotation_matrices = []
        for angle in range(self.num_angles):
            theta = angle * np.pi / self.num_angles
            rot_mat = torch.tensor([[np.cos(theta), np.sin(theta), 0],
                                    [-np.sin(theta), np.cos(theta), 0]], dtype=torch.float)
            rotation_matrices.append(rot_mat)
        rotation_matrices = torch.stack(rotation_matrices)
        return affine_grid(rotation_matrices, [self.num_angles, 1, self.size, self.size], align_corners=True)

    def scale2size(self, scale: float) -> int:
        return int(self.size / 2 ** scale)

    def to(self, device: torch.device):
        self.main_block = torch.nn.Parameter(self.main_block.to(device), requires_grad=True)
        self.zero_freq = torch.nn.Parameter(self.zero_freq.to(device), requires_grad=True)
        # self.nyquist_freq = torch.nn.Parameter(self.nyquist_freq.to(device), requires_grad=True)
        self.nyquist_freq = self.nyquist_freq.to(device)
        self.rotation_grid = self.rotation_grid.to(device)

    def forward(self):
        pass
import numpy as np
import torch
from torch.fft import fft2, ifft2

import scattering_transform.scattering_transform
from scattering_transform.wavelet_models import Wavelet


def scattering_operation(input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
    """
    Convolves and takes the absolute value of two input fields
    :param input_a: the first field in Fourier space (usually a physical field)
    :param input_b: the second field in Fourier space (usually a wavelet filter)
    :return: the output scattering field
    """
    return ifft2(input_a * input_b).abs()


def cross_scattering_operation(input_a: torch.Tensor, input_b: torch.Tensor, common: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(scattering_operation(input_a, common) * scattering_operation(input_b, common))


def clip_fourier_field(field: torch.Tensor, final_size: int) -> torch.Tensor:
    """
    Performs a high frequency clip on a 2D field in Fourier space for a filter scale J.
    :param field: The input field.
    :param final_size: The final size
    :return: the field after clipping
    """
    cl = final_size // 2
    result = torch.cat((torch.cat((field[..., :cl, :cl], field[..., -cl:, :cl]), -2),
                        torch.cat((field[..., :cl, -cl:], field[..., -cl:, -cl:]), -2)
                        ), -1)
    return result


def scale_to_clip_size(scale: int, start_size: int):
    """
    Sets the clip size (and limit to that) that is used to speed up the scattering transform. This is a tradeoff
    between reducing runtime and reducing accuracy.
    :param scale: The wavelet scale factor (often denoted j)
    :param start_size: The size of the initial field
    :return:
    """
    return min(max(int(start_size * 2 ** (-scale + 1)), 32), start_size)


def reduce_coefficients(s0, s1, s2, reduction='rot_avg', normalise_s1=False, normalise_s2=False):

    assert reduction in [None, 'rot_avg', 'ang_avg'], \
        "Wrong output type: must be one of [None, 'rot_avg', 'ang_avg']"

    # Normalisation
    if normalise_s2:
        s2 /= s1[:, :, :, None, None]

    if normalise_s1:
        s1 /= s0[:, :, None]

    # Reduction by averaging
    scale_idx = torch.triu_indices(s1.shape[1], s1.shape[1], offset=1)
    s2 = s2[:, scale_idx[0], :, scale_idx[1]].swapaxes(0, 1)

    if reduction is None:
        s1 = s1.flatten(1, 2)
        s2 = s2.flatten(1, 3)

    elif reduction == 'rot_avg':
        s1 = s1.mean(-1)
        s2 = s2.mean(dim=(-2, -1))

    elif reduction == 'ang_avg':
        s1 = s1.flatten(1, 2)
        num_angles = s2.shape[-1]
        d = torch.abs(torch.arange(num_angles)[:, None] - torch.arange(num_angles)[None, :])
        angle_bins = torch.min(num_angles - d, d)

        s2_vals = torch.zeros((s2.shape[0], s2.shape[1], num_angles // 2), device=s2.device)
        for i in range(num_angles // 2):
            idx = torch.where(angle_bins == i)
            s2_vals[:, :, i] = s2[:, :, idx[0], idx[1]].mean(-1)
        s2 = s2_vals.flatten(1, 2)

    return torch.cat((s0, s1, s2), dim=1)


class ScatteringTransform2d(object):

    def __init__(self, filters: Wavelet):
        super(ScatteringTransform2d, self).__init__()
        self.filters = filters
        self.device = torch.device('cpu')

        # Pre-clip the filters to speed up calculations
        self.clip_sizes = []
        self.filters_clipped = []
        self.clip_scaling_factors = []
        for j in range(self.filters.num_scales):
            clip_size = scale_to_clip_size(j, self.filters.size)
            self.clip_sizes.append(clip_size)
            self.filters_clipped.append(clip_fourier_field(filters.filter_tensor[j], clip_size))
            self.clip_scaling_factors.append(clip_size ** 2 / self.filters.size ** 2)

        self.clip_scaling_factors = torch.tensor(self.clip_scaling_factors)

    def to(self, device):
        self.filters.to(device)
        for j in range(self.filters.num_scales):
            self.filters_clipped[j] = self.filters_clipped[j].to(device)
        self.device = device
        self.clip_scaling_factors = self.clip_scaling_factors.to(device)

    def scattering_transform(self, fields):
        fields_k = fft2(fields)

        coeffs_0 = torch.mean(fields, dim=(-2, -1)).unsqueeze(1)
        coeffs_1, outputs_1 = self._first_order(scattering_operation, [fields_k, ])

        inputs_2 = [fft2(i) for i in outputs_1]
        coeffs_2 = self._second_order(scattering_operation, [inputs_2, ])

        coeffs_1, coeffs_2 = self._rescale_coeffs(coeffs_1, coeffs_2)
        return coeffs_0, coeffs_1, coeffs_2

    def cross_scattering_transform(self, fields_a, fields_b):
        fields_a_k = fft2(fields_a)
        fields_b_k = fft2(fields_b)

        coeffs_0 = torch.mean(fields_a.sqrt() * fields_b.sqrt(), dim=(-2, -1)).unsqueeze(1)  # maybe?
        coeffs_1, outputs_1 = self._first_order(cross_scattering_operation, [fields_a_k, fields_b_k, ])

        inputs_2 = [fft2(i) for i in outputs_1]
        coeffs_2 = self._second_order(scattering_operation, [inputs_2, ])

        coeffs_1, coeffs_2 = self._rescale_coeffs(coeffs_1, coeffs_2)
        return coeffs_0, coeffs_1, coeffs_2

    def _first_order(self, func, input_fields):
        all_output = []
        coeffs = torch.zeros((input_fields[0].shape[0], self.filters.num_scales, self.filters.num_angles),
                             device=input_fields[0].device)
        for scale in range(self.filters.num_scales):
            in_fields = [clip_fourier_field(a, self.clip_sizes[scale])[:, None, ...] for a in input_fields]
            output = func(*in_fields, self.filters_clipped[scale][None, ...])
            coeffs[:, scale] = output.mean((-2, -1))
            all_output.append(output)

        return coeffs, all_output

    def _second_order(self, func, input_fields):

        coeffs = torch.zeros((input_fields[0][0].shape[0], self.filters.num_scales, self.filters.num_angles,
                              self.filters.num_scales, self.filters.num_angles), device=input_fields[0][0].device)

        for scale_1 in range(self.filters.num_scales):
            for scale_2 in range(self.filters.num_scales):
                if scale_2 > scale_1:
                    in_fields = [clip_fourier_field(a[scale_1], self.clip_sizes[scale_2])[:, :, None, ...] for a in
                                 input_fields]
                    output_2 = func(*in_fields, self.filters_clipped[scale_2][None, None, ...])
                    coeffs[:, scale_1, :, scale_2, :] = output_2.mean((-2, -1))

        return coeffs

    def _rescale_coeffs(self, c1, c2):
        c1 = c1 * self.clip_scaling_factors[None, :, None]
        c2 = c2 * self.clip_scaling_factors[None, None, None, :, None]
        return c1, c2

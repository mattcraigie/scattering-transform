import numpy as np
import torch
from torch.fft import fft2, ifft2
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

    assert reduction in [None, 'rot_avg', 'angle_avg'], \
        "Wrong output type: must be one of [None, 'rot_avg', 'angle_avg']"

    # Normalisation
    if normalise_s2:
        s2 /= s1[:, :, :, None, None]

    if normalise_s1:
        s1 /= s0[:, None, None]

    # Reduction by averaging
    s0 = s0.unsqueeze(1)
    scale_idx = torch.triu_indices(s1.shape[-1], s1.shape[-1], offset=1)
    s2 = s2[:, scale_idx[0], :, scale_idx[1]].swapaxes(0, 1)

    if reduction is None:
        s1 = s1.flatten(1, 2)
        s2 = s2.flatten(1, 3)

    elif reduction == 'rot_avg':
        s1 = s1.mean(-1)
        s2 = s2.mean(dim=(-2, -1))

    elif reduction == 'angle_avg':
        raise NotImplementedError  # todo: angle averaging neatly

    return torch.cat((s0, s1, s2), dim=1)


class ScatteringTransform2d(object):

    def __init__(self, filters: Wavelet):
        super(ScatteringTransform2d, self).__init__()
        self.filters = filters
        self.device = torch.device('cpu')

        # Pre-clip the filters to speed up calculations
        self.clip_sizes = []
        self.filters_clipped = []
        for j in range(self.filters.num_scales):
            clip_size = scale_to_clip_size(j, self.filters.size)
            self.clip_sizes.append(clip_size)
            self.filters_clipped.append(clip_fourier_field(filters.filter_tensor[j], clip_size))

        # todo: correctly apply before/after cut scale factor

    def to(self, device):
        self.filters.to(device)
        for j in self.filters.num_scales:
            self.filters_clipped[j].to(device)
        self.device = device

    def scattering_transform(self, fields):
        batch_size = fields.shape[0]

        # ~~~ Zeroth Order ~~~ #
        coeffs_0 = torch.mean(fields, dim=(-2, -1))

        # ~~~ First Order ~~~ #
        fields_k = fft2(fields)
        fields_clipper = lambda j: clip_fourier_field(fields_k, self.clip_sizes[j])
        func_first = lambda j: self._scatt_op(fields_clipper(j), j,
                                              slice_x=(slice(None), None, ...),
                                              slice_filter=(None, ...))
        coeffs_1, outputs_1 = self._first_order(func_first, batch_size)

        # ~~~ Second Order ~~~ #
        outputs_1_clipper = lambda j1, j2: clip_fourier_field(fft2(outputs_1[j1]), self.clip_sizes[j2])
        func_second = lambda j1, j2: self._scatt_op(outputs_1_clipper(j1, j2), j2,
                                                    slice_x=(slice(None), slice(None), None, ...),
                                                    slice_filter=(None, None, ...))
        coeffs_2 = self._second_order(func_second, batch_size)

        return coeffs_0, coeffs_1, coeffs_2

    def cross_scattering_transform(self, fields_a, fields_b):
        batch_size = fields_a.shape[0]

        # ~~~ Zeroth Order ~~~ #
        coeffs_0 = torch.mean(fields_a.sqrt() * fields_b.sqrt(), dim=(-2, -1))   # maybe?

        # ~~~ First Order ~~~ #
        fields_a_k = fft2(fields_a)
        fields_b_k = fft2(fields_b)
        fields_clipper = lambda x, j: clip_fourier_field(x, self.clip_sizes[j])
        func_first = lambda j: self._cross_op(fields_clipper(fields_a_k, j), fields_clipper(fields_b_k, j), j,
                                              slice_xy=(slice(None), None, ...),
                                              slice_filter=(None, ...))
        coeffs_1, outputs_1 = self._first_order(func_first, batch_size)

        # ~~~ Second Order ~~~ #
        outputs_1_clipper = lambda j1, j2: clip_fourier_field(fft2(outputs_1[j1]), self.clip_sizes[j2])
        func_second = lambda j1, j2: self._scatt_op(outputs_1_clipper(j1, j2), j2,
                                                    slice_x=(slice(None), slice(None), None, ...),
                                                    slice_filter=(None, None, ...))
        coeffs_2 = self._second_order(func_second, batch_size)

        return coeffs_0, coeffs_1, coeffs_2

    def _scatt_op(self, x, scale, slice_x=(), slice_filter=()):
        return scattering_operation(x[slice_x], self.filters_clipped[scale][slice_filter])

    def _cross_op(self, x, y, scale, slice_xy=(), slice_filter=()):
        return cross_scattering_operation(x[slice_xy], y[slice_xy],
                                          self.filters_clipped[scale][slice_filter])

    def _first_order(self, func, batch_size):
        all_output = []

        coeffs = torch.zeros((batch_size, self.filters.num_scales, self.filters.num_scales))

        for scale in range(self.filters.num_scales):
            output = func(scale)
            coeffs[:, scale] = output.mean((-2, -1))
            all_output.append(output)

        return coeffs, all_output

    def _second_order(self, func, batch_size):
        coeffs = torch.zeros((batch_size, self.filters.num_scales, self.filters.num_scales,
                                self.filters.num_angles, self.filters.num_angles))

        for scale_1 in range(self.filters.num_scales):
            for scale_2 in range(self.filters.num_scales):
                if scale_2 > scale_1:
                    output_2 = func(scale_1, scale_2)
                    coeffs[:, scale_1, :, scale_2, :] = output_2.mean((-2, -1))

        return coeffs


import torch
from scattering_transform.wavelet_models import Morlet
from scattering_transform.scattering_transform import ScatteringTransform2d, reduce_coefficients
wv = Morlet(128, 6, 6)

st = ScatteringTransform2d(wv)
res = st.cross_scattering_transform(torch.rand(16, 128, 128), torch.rand(16, 128, 128))
print(res[1][0])
red = reduce_coefficients(*res)
print(red[0])





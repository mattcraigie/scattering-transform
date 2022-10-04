import numpy as np
import torch
from torch.fft import fft2, ifft2
from scattering_transform.wavelet_models import Wavelet


def scattering_operation(input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
    """
    Convolves and takes the absolute value of two input fields
    :param input_a: the first field in Fourier space (usually a physical field)
    :param input_b: the second field in Fourier space (usually a wavelet filter)
    :return: the otuput scattering field
    """
    return ifft2(input_a * input_b).abs()


def clip_fourier_field(field: torch.Tensor, final_size: int):
    """
    Performs a high frequency clip on a 2D field in Fourier space for a filter scale J.
    :param field: The input field.
    :param filter_scale: The filter scale
    :return:
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
    return max(int(start_size * 2 ** -scale), 32)


class ScatteringTransform2d(object):

    def __init__(self, filters: Wavelet):
        super(ScatteringTransform2d, self).__init__()
        self.filters = filters
        self.filters_clip = [clip_fourier_field(filters.filter_tensor[j], scale_to_clip_size(j, self.filters.size))
                             for j in range(self.filters.num_scales)]

    def run(self, fields: torch.Tensor, output_type='rot_avg', normalise_s1=False, normalise_s2=False):
        """
        Run the scattering transform for the input fields. It already has wavelets setup.
        :param fields: The fields to run the scattering transform on with Size(batch_size, size, size)
        :return: scattering coefficients
        """
        batch_size = fields.shape[0]
        coeffs_0 = torch.mean(fields, dim=(-2, -1))
        coeffs_1 = torch.zeros(size=(batch_size, self.filters.num_scales, self.filters.num_angles))
        coeffs_2 = torch.zeros((batch_size, self.filters.num_scales, self.filters.num_scales,
                                self.filters.num_angles, self.filters.num_angles))

        fields_k = torch.fft.fft2(fields)

        for scale_1 in range(self.filters.num_scales):
            fields_k_clip = clip_fourier_field(fields_k, scale_to_clip_size(scale_1, self.filters.size))
            scatt_fields_1 = scattering_operation(fields_k_clip.unsqueeze(1), self.filters_clip[scale_1].unsqueeze(0))
            coeffs_1[:, scale_1] = torch.mean(scatt_fields_1, dim=(-2, -1))  # * cut_factor
            scatt_fields_1_k = torch.fft.fft2(scatt_fields_1)

            for scale_2 in range(self.filters.num_scales):
                if scale_2 > scale_1:
                    scatt_fields_1_k_clip = clip_fourier_field(scatt_fields_1_k,
                                                               scale_to_clip_size(scale_2, self.filters.size))
                    scatt_fields_2 = scattering_operation(scatt_fields_1_k_clip.unsqueeze(2),
                                                          self.filters_clip[scale_2][None, None, ...])
                    coeffs_2[:, scale_1, :, scale_2, :] = torch.mean(scatt_fields_2, dim=(-2, -1))  # * cut_factor

        return self.reduce_coeffs(coeffs_0, coeffs_1, coeffs_2, output_type, normalise_s1, normalise_s2)

    def reduce_coeffs(self, s0, s1, s2, output_type='rot_avg', normalise_s1=False, normalise_s2=False):

        assert output_type in ['all', 'rot_avg', 'angle_avg'], \
            "Wrong output type: must be one of ['all', 'rot_avg', 'angle_avg']"

        # Normalisation
        if normalise_s2:
            s2 /= s1[:, :, :, None, None]

        if normalise_s1:
            s1 /= s0[:, None, None]

        # Reduction by averaging
        s0 = s0.unsqueeze(1)
        scale_idx = torch.triu_indices(self.filters.num_scales, self.filters.num_scales, offset=1)
        s2 = s2[:, scale_idx[0], :, scale_idx[1]].swapaxes(0, 1)

        if output_type == 'all':
            s1 = s1.flatten(1, 2)
            s2 = s2.flatten(1, 3)

        elif output_type == 'rot_avg':
            s1 = s1.mean(-1)
            s2 = s2.mean(dim=(-2, -1))

        elif output_type == 'angle_avg':
            raise NotImplementedError  # todo: angle averaging neatly

        # todo: correctly apply before/after cut scale factor

        return torch.cat((s0, s1, s2), dim=1)


class CrossScatteringTransform2d(ScatteringTransform2d):
    def __init__(self, filters: Wavelet):
        super(CrossScatteringTransform2d, self).__init__(filters)

    def run_standard(self, fields: torch.Tensor, output_type='rot_avg'):
        return self.run(fields, output_type)

    def run_cross(self, fields_a: torch.Tensor, fields_b: torch.Tensor, output_type='rot_avg', normalise_s1=False,
                  normalise_s2=False):
        """
        Run the cross scattering transform for the input fields. It already has wavelets setup.

        Couple of different ways we can implement the second order.
            - We could reiterate the scattering operation on the cross fields
            - We could cross the 2nd order fields for each
        I think the first option holds more info but I'll have to test the second.


        :param fields_a: The first set of fields to run the scattering transform on with Size(batch_size, size, size)
        :param fields_b: The second set of fields to run the scattering transform on with Size(batch_size, size, size)
        :return: scattering coefficients
        """
        assert fields_a.shape == fields_b.shape, \
            "Fields must match shape but have shapes {} and {}".format(fields_a.shape, fields_b.shape)

        batch_size = fields_a.shape[0]
        coeffs_0 = torch.mean(torch.sqrt(fields_a * fields_b), dim=(-2, -1))
        coeffs_1 = torch.zeros(size=(batch_size, self.filters.num_scales, self.filters.num_angles))
        coeffs_2 = torch.zeros((batch_size, self.filters.num_scales, self.filters.num_scales,
                                self.filters.num_angles, self.filters.num_angles))

        fields_a_k = torch.fft.fft2(fields_a)
        fields_b_k = torch.fft.fft2(fields_a)

        for scale_1 in range(self.filters.num_scales):
            fields_a_k_clip = clip_fourier_field(fields_a_k, scale_1)
            fields_b_k_clip = clip_fourier_field(fields_b_k, scale_1)
            cross_scatt_fields_1 = self.cross_scattering(fields_a_k_clip.unsqueeze(1),
                                                         fields_b_k_clip.unsqueeze(1),
                                                         self.filters_clip[scale_1].unsqueeze(0))
            coeffs_1[:, scale_1] = torch.mean(cross_scatt_fields_1, dim=(-2, -1))  # * cut_factor

            scatt_fields_1_k = torch.fft.fft2(cross_scatt_fields_1)

            for scale_2 in range(self.filters.num_scales):
                if scale_2 > scale_1:
                    scatt_fields_1_k_clip = clip_fourier_field(scatt_fields_1_k, scale_2)
                    scatt_fields_2 = scattering_operation(scatt_fields_1_k_clip.unsqueeze(2),
                                                          self.filters_clip[scale_2][None, None, ...])
                    coeffs_2[:, scale_1, :, scale_2, :] = torch.mean(scatt_fields_2, dim=(-2, -1))  # * cut_factor


        return self.reduce_coeffs(coeffs_0, coeffs_1, coeffs_2, output_type, normalise_s1, normalise_s2)

    def cross_scattering(self, a, b, psi):
        return torch.sqrt(ifft2(a * psi).abs() * ifft2(b * psi).abs())




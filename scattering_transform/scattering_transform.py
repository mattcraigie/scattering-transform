import numpy as np
import torch
from torch.fft import fft2, ifft2
from scattering_transform.filters import FilterBank


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
    assert final_size <= field.shape[-1], "Final size must be smaller than or same as the input field size"
    cl = final_size // 2
    result = torch.cat((torch.cat((field[..., :cl, :cl], field[..., -cl:, :cl]), -2),
                        torch.cat((field[..., :cl, -cl:], field[..., -cl:, -cl:]), -2)
                        ), -1)
    return result


def scale_to_clip_size(scale: int, start_size: int):
    """
    Returns the default clip size that is used to speed up the scattering transform. Works with the Morlet wavelets.
    :param scale: The wavelet scale factor (often denoted j)
    :param start_size: The size of the initial field
    :return:
    """
    return min(max(int(start_size * 2 ** (-scale + 1)), 32), start_size)


# legacy function, class use is preferred
def reduce_coefficients(s0, s1, s2, reduction='rot_avg', normalise_s1=False, normalise_s2=False):

    assert reduction in [None, 'rot_avg', 'ang_avg'], \
        "Wrong output type: must be one of [None, 'rot_avg', 'ang_avg']"

    # Normalisation
    if normalise_s2:
        s2 = s2 / s1[:, :, :, None, None]

    if normalise_s1:
        s1 = s1 / s0[:, :, None]

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
        s1 = s1.mean(-1)
        num_angles = s2.shape[-1]
        d = torch.abs(torch.arange(num_angles)[:, None] - torch.arange(num_angles)[None, :])
        angle_bins = torch.min(num_angles - d, d)
        num_bins = torch.unique(angle_bins).shape[0]
        print(s2.shape)
        s2_vals = torch.zeros((s2.shape[0], s2.shape[1], num_bins), device=s2.device)
        for i in range(num_bins):
            idx = torch.where(angle_bins == i)
            print(s2_vals[:, :, i].shape)
            print(s2[:, :, idx[0], idx[1]].mean(-1).shape)
            s2_vals[:, :, i] = s2[:, :, idx[0], idx[1]].mean(-1)
        s2 = s2_vals.flatten(1, 2)

    return torch.cat((s0, s1, s2), dim=1)


class Reducer(torch.nn.Module):

    # todo: add in a method that takes in an index and outputs the J1, L1, J2, L2 etc that it corresponds to.

    def __init__(self, filters, reduction, normalise_s2=False):
        """Class that reduces the scattering coefficients based on the symmetry in the field. It also performs some
        normalisation. The main reason to use a class rather than a function is so that we can pre-calculate the
        number of scattering coefficients, given the reduction.

        Note that the first order coefficients are independent of the mean of the field, since by the admissibility
        criterion, the zero-freq component of the filter is always zero (i.e. any info about the mean is destroyed).
        Therefore, normalising by dividing out S0 always has the effect of correlating, instead of de-correlating.
        Which is why I've removed it as an option.

        An alternative to the normalisation that I could implement in the future, is to set a normalisation factor such
        that instances of a GRF of that size would have standard normally distributed coefficients. The goal being that
        these outputs will then be distributed approximately between (-3, 3) and therefore be well-behaved as inputs
        to neural networks, without requiring batch normalisation. An added benefit is that it will be immediately clear
        whether the field has more or less `scattering power' through that filter.
        """
        super(Reducer, self).__init__()

        assert reduction in [None, 'none', 'rot_avg', 'ang_avg', 'asymm_ang_avg'], \
            "Wrong reduction: must be one of [None (or 'none'), 'rot_avg', 'ang_avg', 'asymm_ang_avg']"

        assert filters.num_angles % 2 == 0, "Number of angles must be even for now."

        self.reduction = reduction
        self.normalise_s2 = normalise_s2

        # run a test field through the scattering transform to work out the required number of outputs
        test_field = torch.randn(1, filters.size, filters.size).to(filters.device)
        st = ScatteringTransform2d(filters)
        st.to(filters.device)
        s = st.forward(test_field)
        self.num_outputs = self.forward(s).shape[-1]

    def forward(self, s):

        s0, s1, s2 = s

        # shorthands for cleaner code
        sln = slice(None)  # SLice None, i.e. :
        batch_dims = s0.ndim - 1
        batch_sizes = list(s0.shape[:-1])
        bds = [sln]*batch_dims  # Batch Dim Slice nones

        # Normalisation
        if self.normalise_s2:
            s2 = s2 / s1[bds + [sln, sln, None, None]]

        # Selecting only j2 > j1 entries
        scale_idx = torch.triu_indices(s1.shape[-2], s1.shape[-2], offset=1)
        s2 = s2[bds + [scale_idx[0], sln, scale_idx[1]]]
        s2 = s2.permute([i + 1 for i in range(batch_dims)] + [0, s2.ndim-2, s2.ndim-1])

        if self.reduction is None or self.reduction == 'none':
            s1 = s1.flatten(-2, -1)
            s2 = s2.flatten(-3, -1)

        elif self.reduction == 'rot_avg':
            s1 = s1.mean(-1)
            s2 = s2.mean(dim=(-2, -1))

        elif self.reduction == 'ang_avg':
            s1 = s1.mean(-1)
            num_angles = s2.shape[-1]

            # Calculate the distance between each angle in a 2D tensor. Angles left are equivalent to angles right,
            # which is why we take the absolute value.
            delta = torch.abs(torch.arange(num_angles)[:, None] - torch.arange(num_angles)[None, :])

            # Use the rotational symmetry of the filters (e.g. for J=4, 3 angles apart is equivalent to 1 angle apart,
            # when you got the other way, so we can just take the min of the two).
            delta = torch.min(num_angles - delta, delta)

            s2_vals_all = []
            for i in range(delta.min(), delta.max()+1):
                idx = torch.where(delta == i)  # find the indices of the angles that are i apart
                s2_vals_all.append(s2[bds + [sln, idx[0], idx[1]]].mean(-1))  # take the average of everything with the same angle difference
            s2 = torch.cat(s2_vals_all, dim=-1)

        elif self.reduction == 'asymm_ang_avg':
            s1 = s1.mean(-1)
            num_angles = s2.shape[-1]

            # calculate the distance between each angle in a 2D tensor. Due to asymmetry, angles left are not equivalent
            # to angles right.
            delta = torch.arange(num_angles)[:, None] - torch.arange(num_angles)[None, :]

            s2_vals_all = []
            for i in range(delta.min(), delta.max()+1):
                idx = torch.where(delta == i)  # find the indices of the angles that are i apart
                part = s2[bds + [sln, idx[0], idx[1]]]
                s2_vals_all.append(part.mean(-1))
            s2 = torch.cat(s2_vals_all, dim=-1)

        return torch.cat((s0, s1, s2), dim=batch_dims)


class ScatteringTransform2d(torch.nn.Module):

    # todo: change the code to take in cross_dim on the forward pass and have the option to cross between channels in
    #       that dim

    def __init__(self, filters: FilterBank, clip_sizes=None):
        super(ScatteringTransform2d, self).__init__()
        self.filters = filters
        self.device = torch.device('cpu')

        # We gain a significant speedup by clipping the filters to only the relevant fourier modes
        if clip_sizes is None:
            self.clip_sizes = []
            for j in range(self.filters.num_scales):
                clip_size = scale_to_clip_size(j, self.filters.size)
                self.clip_sizes.append(clip_size)
        else:
            assert len(clip_sizes) == filters.num_scales, "Clip sizes must match the number of scales"
            self.clip_sizes = clip_sizes

        # Storing the clip scaling factors, since we must use these to correctly scale the scattering fields.
        self.clip_scaling_factors = []
        for j in range(self.filters.num_scales):
            self.clip_scaling_factors.append(self.clip_sizes[j] ** 2 / self.filters.size ** 2)
        self.clip_scaling_factors = torch.tensor(self.clip_scaling_factors)

        # Clip the filters to speed up calculations
        self.clip_filters()

    def to(self, device):
        self.device = device
        self.filters.to(device)
        for j in range(self.filters.num_scales):
            self.filters_clipped[j] = self.filters_clipped[j].to(device)
        self.clip_scaling_factors = self.clip_scaling_factors.to(device)

    def clip_filters(self):
        self.filters_clipped = []
        for j in range(self.filters.num_scales):
            self.filters_clipped.append(
                clip_fourier_field(self.filters.filter_tensor[j], self.clip_sizes[j])
            )

    def scattering_transform(self, fields, pre_fft=False):
        if len(fields.shape) < 3:
            raise ValueError("The input fields should have shape (batch, (...,) size, size)")

        batch_sizes = fields.shape[:fields.ndim-2]

        if pre_fft:
            fields_k = fields
        else:
            fields_k = fft2(fields)

        coeffs_0 = torch.mean(fields, dim=(-2, -1)).unsqueeze(-1)
        coeffs_1, outputs_1 = self._first_order(scattering_operation, [fields_k, ], batch_sizes)

        inputs_2 = [fft2(i) for i in outputs_1]
        coeffs_2 = self._second_order(scattering_operation, [inputs_2, ], batch_sizes)

        coeffs_1, coeffs_2 = self._rescale_coeffs(coeffs_1, coeffs_2, len(batch_sizes))
        return coeffs_0, coeffs_1, coeffs_2

    def _first_order(self, func, input_fields, batch_sizes):
        all_output = []
        coeffs = torch.zeros(batch_sizes + (self.filters.num_scales, self.filters.num_angles),
                             device=input_fields[0].device)  # should I replace with cat/stack?
        for scale in range(self.filters.num_scales):
            in_fields = [clip_fourier_field(a, self.clip_sizes[scale])[..., None, :, :] for a in input_fields]
            output = func(*in_fields, self.filters_clipped[scale][[None]*len(batch_sizes)+[Ellipsis]])
            coeffs[[slice(None)]*len(batch_sizes)+[scale]] = output.mean((-2, -1))
            all_output.append(output)

        return coeffs, all_output

    def _second_order(self, func, input_fields, batch_sizes):

        coeffs = torch.zeros(batch_sizes + (self.filters.num_scales, self.filters.num_angles,
                              self.filters.num_scales, self.filters.num_angles), device=input_fields[0][0].device)

        for scale_1 in range(self.filters.num_scales):
            for scale_2 in range(self.filters.num_scales):
                if scale_2 > scale_1:
                    in_fields = [clip_fourier_field(a[scale_1], self.clip_sizes[scale_2])[..., None, :, :] for a in
                                 input_fields]
                    output_2 = func(*in_fields, self.filters_clipped[scale_2][[None]*(len(batch_sizes)+1)+[Ellipsis]])
                    coeffs[[slice(None)]*len(batch_sizes)+[scale_1, slice(None), scale_2, slice(None)]] = output_2.mean((-2, -1))

        return coeffs

    # def cross_scattering_transform(self, fields_a, fields_b):
    #     fields_a_k = fft2(fields_a)
    #     fields_b_k = fft2(fields_b)
    #
    #     coeffs_0 = torch.mean(fields_a.sqrt() * fields_b.sqrt(), dim=(-2, -1)).unsqueeze(1)  # maybe?
    #     coeffs_1, outputs_1 = self._first_order(cross_scattering_operation, [fields_a_k, fields_b_k, ])
    #
    #     inputs_2 = [fft2(i) for i in outputs_1]
    #     coeffs_2 = self._second_order(scattering_operation, [inputs_2, ])
    #
    #     coeffs_1, coeffs_2 = self._rescale_coeffs(coeffs_1, coeffs_2)
    #     return coeffs_0, coeffs_1, coeffs_2

    def _rescale_coeffs(self, c1, c2, batch_dim):
        c1 = c1 * self.clip_scaling_factors[[None] * batch_dim + [slice(None), None]]
        c2 = c2 * self.clip_scaling_factors[[None] * batch_dim + [None, None, slice(None), None]]
        return c1, c2

    def forward(self, fields, pre_fft=False, cross_dim=None):
        if cross_dim is None:
            return self.scattering_transform(fields, pre_fft=pre_fft)
        else:
            raise NotImplementedError("Cross scattering not implemented yet")
import torch
from torch.fft import fft2, ifft2, fftn, ifftn
from .filters import FilterBank
from .general_functions import clip_fourier_field


def scattering_operation(input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
    """
    Convolves and takes the absolute value of two input fields
    :param input_a: the first field in Fourier space (usually a physical field)
    :param input_b: the second field in Fourier space (usually a wavelet filter)
    :return: the output scattering field
    """
    return ifft2(input_a * input_b).abs()




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
        test_field = torch.randn(1, 1, filters.size, filters.size).to(filters.device)
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
    def __init__(self, filter_bank: FilterBank):
        super(ScatteringTransform2d, self).__init__()

        # check the filter bank has been correctly specified
        assert getattr(filter_bank, "clip_sizes", None) is not None, "Clip sizes must be specified for the filter bank"
        assert getattr(filter_bank, "filters", None) is not None, "Filters must be specified for the filter bank"
        self.filter_bank = filter_bank

        # add the clip scaling factors - these account for the effects of clipping on the scattering coefficients
        self.clip_scaling_factors = [self.filter_bank.clip_sizes[j] ** 2 / self.filter_bank.size ** 2
                                     for j in range(self.filter_bank.num_scales)]
        self.clip_scaling_factors = torch.tensor(self.clip_scaling_factors)

    def scattering_transform(self, fields):
        # fields should be a tensor of shape (batch, channels, size, size)
        assert len(fields.shape) == 4, "The input fields should have shape (batch, channels, size, size)"

        # the zeroth order coefficient
        coeffs_0 = torch.mean(fields, dim=(-2, -1)).unsqueeze(-1)
        coeffs_1, fields_1 = self._first_order(fields)
        coeffs_2 = self._second_order(fields_1)

        # rescale the coefficients to account for clipping
        coeffs_1, coeffs_2 = self._rescale_coeffs(coeffs_1, coeffs_2)

        return coeffs_0, coeffs_1, coeffs_2

    def _first_order(self, fields):

        fields_fourier = fft2(fields)

        first_order_coefficients = []
        first_order_scattering_fields = []
        for scale in range(self.filter_bank.num_scales):

            # clip the fields to the correct size
            clip_size = self.filter_bank.clip_sizes[scale]
            fields_fourier_clipped = clip_fourier_field(fields_fourier, clip_size)

            # get the filter for this scale
            scale_filter = self.filter_bank.filters[scale]

            # add broadcasting dimensions
            fields_fourier_clipped = fields_fourier_clipped[:, :, None, ...]
            scale_filter = scale_filter[None, None, ...]

            # apply the scattering operation
            scattering_fields = scattering_operation(fields_fourier_clipped, scale_filter)
            first_order_scattering_fields.append(scattering_fields)

            # compute the mean of the scattering fields to get the first order coefficients
            coefficients = scattering_fields.mean((-2, -1))
            first_order_coefficients.append(coefficients)

        # stack the coefficients along the scale dimension
        first_order_coefficients = torch.stack(first_order_coefficients, dim=2)  # shape (batch, channels, scales, angles)
        return first_order_coefficients, first_order_scattering_fields

    def _second_order(self, field_list):

        second_order_coefficients = []
        for scale_1 in range(self.filter_bank.num_scales):
            for scale_2 in range(self.filter_bank.num_scales):
                if scale_2 > scale_1:

                    fields_fourier = fft2(field_list[scale_1])

                    # clip the fields to the correct size
                    clip_size = self.filter_bank.clip_sizes[scale_2]
                    fields_fourier_clipped = clip_fourier_field(fields_fourier, clip_size)

                    # get the filter for this scale
                    scale_filter = self.filter_bank.filters[scale_2]

                    # add broadcasting dimensions
                    fields_fourier_clipped = fields_fourier_clipped[..., None, :, :]
                    scale_filter = scale_filter[None, None, None, ...]

                    # apply the scattering operation to the first order fields
                    scattering_fields = scattering_operation(fields_fourier_clipped, scale_filter)

                    # compute the mean of the scattering fields to get the second order coefficients
                    coefficients = scattering_fields.mean((-2, -1))
                    second_order_coefficients.append(coefficients)
                else:
                    # if scale_2 < scale_1 there is no useful information, do not compute and set to zero
                    second_order_coefficients.append(
                        torch.zeros(field_list[0].shape[:2] + (self.filter_bank.num_angles, self.filter_bank.num_angles)
                                    ))

        # arrange the coefficients into a single tensor of shape (batch, channels, scale 1, scale 2, angle 1, angle 2)
        b, c, l1, l2 = second_order_coefficients[0].shape
        j1 = j2 = self.filter_bank.num_scales
        second_order_coefficients = torch.stack(second_order_coefficients, dim=-1)
        second_order_coefficients = second_order_coefficients.reshape(b, c, l1, l2, j1, j2)
        second_order_coefficients = second_order_coefficients.permute(0, 1, 4, 2, 5, 3)

        return second_order_coefficients

    def _rescale_coeffs(self, c1, c2):

        c1 = c1 * self.clip_scaling_factors[None, None, :, None]
        c2 = c2 * self.clip_scaling_factors[None, None, None, None, :, None]

        return c1, c2

    def forward(self, fields):
        return self.scattering_transform(fields)

    def to(self, device):
        super(ScatteringTransform2d, self).to(device)
        self.filter_bank.to(device)
        self.clip_scaling_factors = self.clip_scaling_factors.to(device)
        return self

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


class ScatteringTransform2d(torch.nn.Module):
    def __init__(self, filter_bank: FilterBank):
        """
        A class to compute the scattering transform of a 2D field, for a given set of filters stored in a FilterBank
        class.

        :param filter_bank: The filter bank to use for the scattering transform.
        """

        super(ScatteringTransform2d, self).__init__()

        # check the filter bank has been correctly specified
        assert getattr(filter_bank, "clip_sizes", None) is not None, "Clip sizes must be specified for the filter bank"
        assert getattr(filter_bank, "filters", None) is not None, "Filters must be specified for the filter bank"
        self.filter_bank = filter_bank

        # add the clip scaling factors - these account for the effects of clipping on the scattering coefficients
        self.clip_scaling_factors = [self.filter_bank.clip_sizes[j] ** 2 / self.filter_bank.size ** 2
                                     for j in range(self.filter_bank.num_scales)]
        self.clip_scaling_factors = torch.tensor(self.clip_scaling_factors)

    def _scattering_transform(self, fields):
        """
        Computes the scattering transform of a batch of 2D fields. Use the forward method of the class to call.
        :param fields: A tensor of shape (batch, channels, size, size) containing the fields to transform.
        :return: A tuple containing the zeroth, first and second order scattering coefficients.
        """

        assert len(fields.shape) == 4, "The input fields should have shape (batch, channels, size, size)"

        # the zeroth order coefficient
        coeffs_0 = torch.mean(fields, dim=(-2, -1)).unsqueeze(-1)
        coeffs_1, fields_1 = self._first_order(fields)
        coeffs_2 = self._second_order(fields_1)

        # rescale the coefficients to account for clipping
        coeffs_1, coeffs_2 = self._rescale_coeffs(coeffs_1, coeffs_2)

        return coeffs_0, coeffs_1, coeffs_2

    def _first_order(self, fields):
        """
        Computes the first order scattering coefficients and the first order scattering fields.
        :param fields: A tensor of shape (batch, channels, size, size) containing the fields to transform.
        :return: A tuple containing the first order scattering coefficients and the first order scattering fields.
        """

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
        """
        Computes the second order scattering coefficients.
        :param field_list: A list of tensors containing the first order scattering fields.
        :return: A tensor containing the second order scattering coefficients.
        """
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
        """
        Rescales the scattering coefficients to account for the effects of clipping.
        :param c1: the first order scattering coefficients
        :param c2: the second order scattering coefficients
        :return: a tuple containing the rescaled coefficients
        """

        c1 = c1 * self.clip_scaling_factors[None, None, :, None]
        c2 = c2 * self.clip_scaling_factors[None, None, None, None, :, None]

        return c1, c2

    def forward(self, fields):
        """
        The forward pass of the scattering transform.
        :param fields: A tensor of shape (batch, channels, size, size) containing the fields to transform.
        :return: A tuple containing the zeroth, first and second order scattering coefficients.
        """
        return self._scattering_transform(fields)

    def to(self, device):
        """
        Moves the scattering transform to the specified device.
        :param device: the device to move to
        :return: self
        """
        super(ScatteringTransform2d, self).to(device)
        self.filter_bank.to(device)
        self.clip_scaling_factors = self.clip_scaling_factors.to(device)
        return self


import torch
from torch.fft import fft2
from .filters_3d import FilterBank3d
from .general_functions import clip_fourier_field_3d, scattering_operation
from .scattering_transform import ScatteringTransform2d


class ScatteringTransform3d(ScatteringTransform2d):
    def __init__(self, filter_bank: FilterBank3d):
        """
        A class to compute the scattering transform of a 2D field, for a given set of filters stored in a FilterBank
        class.

        :param filter_bank: The filter bank to use for the scattering transform.
        """

        super(ScatteringTransform3d, self).__init__(filter_bank)

        # add the clip scaling factors - adjusted for 3D
        self.clip_scaling_factors = [self.filter_bank.clip_sizes[j] ** 3 / self.filter_bank.size ** 3
                                     for j in range(self.filter_bank.num_scales)]
        self.clip_scaling_factors = torch.tensor(self.clip_scaling_factors)

    def _scattering_transform(self, fields):
        """
        Computes the scattering transform of a batch of 2D fields. Use the forward method of the class to call.
        :param fields: A tensor of shape (batch, channels, size, size, size) containing the fields to transform.
        :return: A tuple containing the zeroth, first and second order scattering coefficients.
        """

        assert len(fields.shape) == 5, "The input fields should have shape (batch, channels, size, size, size)"

        # the zeroth order coefficient
        coeffs_0 = torch.mean(fields, dim=(-3, -2, -1)).unsqueeze(-1)
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
            fields_fourier_clipped = clip_fourier_field_3d(fields_fourier, clip_size)

            # get the filter for this scale
            scale_filter = self.filter_bank.filters[scale]

            # add broadcasting dimensions
            fields_fourier_clipped = fields_fourier_clipped[:, :, None, ...]
            scale_filter = scale_filter[None, None, ...]

            # apply the scattering operation
            scattering_fields = scattering_operation(fields_fourier_clipped, scale_filter)
            first_order_scattering_fields.append(scattering_fields)

            # compute the mean of the scattering fields to get the first order coefficients
            coefficients = scattering_fields.mean((-3, -2, -1))
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
                    fields_fourier_clipped = clip_fourier_field_3d(fields_fourier, clip_size)

                    # get the filter for this scale
                    scale_filter = self.filter_bank.filters[scale_2]

                    # add broadcasting dimensions
                    fields_fourier_clipped = fields_fourier_clipped[:, :, :, None, ...]
                    scale_filter = scale_filter[None, None, None, ...]

                    # apply the scattering operation to the first order fields
                    scattering_fields = scattering_operation(fields_fourier_clipped, scale_filter)

                    # compute the mean of the scattering fields to get the second order coefficients
                    coefficients = scattering_fields.mean((-3, -2, -1))
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

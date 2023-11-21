import torch
from torch.fft import fftn
from .filters_3d import FilterBank3d
from .general_functions import clip_fourier_field_3d, scattering_operation_3d
from .scattering_transform import ScatteringTransform2d


class ScatteringTransform3d(ScatteringTransform2d):
    def __init__(self, filter_bank: FilterBank3d):
        """
        A class to compute the scattering transform of a 3D field, for a given set of filters stored in a FilterBank
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
        :param fields: A tensor of shape (batch, channels, size, size, size) containing the fields to transform.
        :return: A tuple containing the first order scattering coefficients and the first order scattering fields.
        """

        fields_fourier = fftn(fields, dim=(-3, -2, -1))

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
            scattering_fields = scattering_operation_3d(fields_fourier_clipped, scale_filter)
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

                    fields_fourier = fftn(field_list[scale_1], dim=(-3, -2, -1))

                    # clip the fields to the correct size
                    clip_size = self.filter_bank.clip_sizes[scale_2]
                    fields_fourier_clipped = clip_fourier_field_3d(fields_fourier, clip_size)

                    # get the filter for this scale
                    scale_filter = self.filter_bank.filters[scale_2]

                    # add broadcasting dimensions
                    fields_fourier_clipped = fields_fourier_clipped[:, :, :, None, ...]
                    scale_filter = scale_filter[None, None, None, ...]

                    # apply the scattering operation to the first order fields
                    scattering_fields = scattering_operation_3d(fields_fourier_clipped, scale_filter)

                    # compute the mean of the scattering fields to get the second order coefficients
                    coefficients = scattering_fields.mean((-3, -2, -1))
                    second_order_coefficients.append(coefficients)
                else:
                    # if scale_2 < scale_1 there is no useful information, do not compute and set to zero
                    second_order_coefficients.append(
                        torch.zeros(field_list[0].shape[:2] + (self.filter_bank.num_angles, self.filter_bank.num_angles)
                                    ).to(self.device))

        # arrange the coefficients into a single tensor of shape (batch, channels, scale 1, angle 1, scale 2, angle 2)
        b, c, l1, l2 = second_order_coefficients[0].shape
        j1 = j2 = self.filter_bank.num_scales
        second_order_coefficients = torch.stack(second_order_coefficients, dim=-1)
        second_order_coefficients = second_order_coefficients.reshape(b, c, l1, l2, j1, j2)
        second_order_coefficients = second_order_coefficients.permute(0, 1, 4, 2, 5, 3)

        return second_order_coefficients


class ThirdOrderScatteringTransform3d(ScatteringTransform2d):
    def __init__(self, filter_bank: FilterBank3d):
        """
        A class to compute the scattering transform of a 3D field, for a given set of filters stored in a FilterBank
        class.

        :param filter_bank: The filter bank to use for the scattering transform.
        """

        super(ThirdOrderScatteringTransform3d, self).__init__(filter_bank)

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
        coeffs_2, fields_2 = self._second_order(fields_1)
        coeffs_3 = self._third_order(fields_2)

        # rescale the coefficients to account for clipping
        coeffs_1, coeffs_2, coeffs_3 = self._rescale_coeffs(coeffs_1, coeffs_2, coeffs_3)

        return coeffs_0, coeffs_1, coeffs_2, coeffs_3

    def _first_order(self, fields):
        """
        Computes the first order scattering coefficients and the first order scattering fields.
        :param fields: A tensor of shape (batch, channels, size, size, size) containing the fields to transform.
        :return: A tuple containing the first order scattering coefficients and the first order scattering fields.
        """

        fields_fourier = fftn(fields, dim=(-3, -2, -1))

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
            scattering_fields = scattering_operation_3d(fields_fourier_clipped, scale_filter)
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
        second_order_fields = []
        for scale_1 in range(self.filter_bank.num_scales):
            for scale_2 in range(self.filter_bank.num_scales):
                if scale_2 > scale_1:

                    fields_fourier = fftn(field_list[scale_1], dim=(-3, -2, -1))

                    # clip the fields to the correct size
                    clip_size = self.filter_bank.clip_sizes[scale_2]
                    fields_fourier_clipped = clip_fourier_field_3d(fields_fourier, clip_size)

                    # get the filter for this scale
                    scale_filter = self.filter_bank.filters[scale_2]

                    # add broadcasting dimensions
                    fields_fourier_clipped = fields_fourier_clipped[:, :, :, None, ...]
                    scale_filter = scale_filter[None, None, None, ...]

                    # apply the scattering operation to the first order fields
                    scattering_fields = scattering_operation_3d(fields_fourier_clipped, scale_filter)

                    # compute the mean of the scattering fields to get the second order coefficients
                    coefficients = scattering_fields.mean((-3, -2, -1))
                    second_order_coefficients.append(coefficients)
                    second_order_fields.append(scattering_fields)
                else:
                    # if scale_2 < scale_1 there is no useful information, do not compute and set to zero
                    second_order_coefficients.append(
                        torch.zeros(field_list[0].shape[:2] + (self.filter_bank.num_angles, self.filter_bank.num_angles)
                                    ).to(self.device))
                    second_order_fields.append(None)

        # arrange the coefficients into a single tensor of shape (batch, channels, scale 1, angle 1, scale 2, angle 2)
        b, c, l1, l2 = second_order_coefficients[0].shape
        j1 = j2 = self.filter_bank.num_scales
        second_order_coefficients = torch.stack(second_order_coefficients, dim=-1)
        second_order_coefficients = second_order_coefficients.reshape(b, c, l1, l2, j1, j2)
        second_order_coefficients = second_order_coefficients.permute(0, 1, 4, 2, 5, 3)

        return second_order_coefficients, second_order_fields

    def _third_order(self, field_list):
        """
        Computes the second order scattering coefficients.
        :param field_list: A list of tensors containing the first order scattering fields.
        :return: A tensor containing the second order scattering coefficients.
        """
        third_order_coefficients = []
        num_scales = self.filter_bank.num_scales

        # when scale_1 < scale_2 or scale_2 < scale_3, the coeffs are zeros
        for i in field_list:
            if i is not None:
                shape = i.shape[:2]
                break
            else:
                shape = None

        assert shape is not None, "All fields are None, cannot compute third order coefficients"

        none_appendage = torch.zeros(shape + (self.filter_bank.num_angles, self.filter_bank.num_angles, self.filter_bank.num_angles)).to(self.device)



        for scale_1 in range(num_scales):
            for scale_2 in range(num_scales):
                for scale_3 in range(num_scales):
                    if scale_1 < scale_2 < scale_3:

                        fields_fourier = fftn(field_list[scale_1 * num_scales + scale_2 % num_scales], dim=(-3, -2, -1))

                        # clip the fields to the correct size
                        clip_size = self.filter_bank.clip_sizes[scale_2]
                        fields_fourier_clipped = clip_fourier_field_3d(fields_fourier, clip_size)

                        # get the filter for this scale
                        scale_filter = self.filter_bank.filters[scale_2]

                        # add broadcasting dimensions
                        fields_fourier_clipped = fields_fourier_clipped[:, :, :, :, None, ...]
                        scale_filter = scale_filter[None, None, None, None, ...]

                        # apply the scattering operation to the first order fields
                        scattering_fields = scattering_operation_3d(fields_fourier_clipped, scale_filter)

                        # compute the mean of the scattering fields to get the second order coefficients
                        coefficients = scattering_fields.mean((-3, -2, -1))
                        third_order_coefficients.append(coefficients)
                    else:
                        # if scale_2 < scale_1 there is no useful information, do not compute and set to zero
                        third_order_coefficients.append(none_appendage)

        # arrange the coefficients into a single tensor of shape (batch, channels, scale 1, angle 1, scale 2, angle 2)
        b, c, l1, l2, l3 = third_order_coefficients[0].shape
        j1 = j2 = j3 = self.filter_bank.num_scales
        third_order_coefficients = torch.stack(third_order_coefficients, dim=-1)
        third_order_coefficients = third_order_coefficients.reshape(b, c, l1, l2, l3, j1, j2, j3)
        third_order_coefficients = third_order_coefficients.permute(0, 1, 5, 2, 6, 3, 7, 4)

        return third_order_coefficients


    def _rescale_coeffs(self, c1, c2, c3):
        """
        Rescales the scattering coefficients to account for the effects of clipping.
        :param c1: the first order scattering coefficients
        :param c2: the second order scattering coefficients
        :return: a tuple containing the rescaled coefficients
        """

        c1 = c1 * self.clip_scaling_factors[None, None, :, None]
        c2 = c2 * self.clip_scaling_factors[None, None, None, None, :, None]
        c3 = c3 * self.clip_scaling_factors[None, None, None, None, None, None, :, None]

        return c1, c2, c3


class FirstOrderScatteringTransform3d(ScatteringTransform2d):
    def __init__(self, filter_bank: FilterBank3d):
        """
        A class to compute the scattering transform of a 3D field, for a given set of filters stored in a FilterBank
        class.

        :param filter_bank: The filter bank to use for the scattering transform.
        """

        super(FirstOrderScatteringTransform3d, self).__init__(filter_bank)

        # add the clip scaling factors - adjusted for 3D
        self.clip_scaling_factors = [self.filter_bank.clip_sizes[j] ** 3 / self.filter_bank.size ** 3
                                     for j in range(self.filter_bank.num_scales)]
        self.clip_scaling_factors = torch.tensor(self.clip_scaling_factors)

    def _scattering_transform(self, fields):

        assert len(fields.shape) == 5, "The input fields should have shape (batch, channels, size, size, size)"

        # the zeroth order coefficient
        coeffs_0 = torch.mean(fields, dim=(-3, -2, -1)).unsqueeze(-1)
        coeffs_1, fields_1 = self._first_order(fields)

        # rescale the coefficients to account for clipping
        coeffs_1 = self._rescale_coeffs(coeffs_1)

        return torch.cat([coeffs_0, coeffs_1], dim=-1)

    def _rescale_coeffs(self, c1):

        return c1 * self.clip_scaling_factors[None, None, :, None]
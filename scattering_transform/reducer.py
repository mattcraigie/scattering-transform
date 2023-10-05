import torch
from .scattering_transform import ScatteringTransform2d
from .scattering_transform_3d import ScatteringTransform3d


class Reducer(torch.nn.Module):

    # todo: add in a method that takes in an index and outputs the J1, L1, J2, L2 etc that it corresponds to.

    def __init__(self, filters, reduction, normalise_s2=False, filters_3d=False):
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
        if filters_3d:
            test_field = torch.randn(1, 1, filters.size, filters.size, filters.size).to(filters.device)
            st = ScatteringTransform3d(filters)
        else:
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
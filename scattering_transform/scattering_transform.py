import numpy as np
import torch
from scattering_transform.wavelet_models import WaveletsMorlet


class ScatteringTransformFull:


    def __init__(self, filters: WaveletsMorlet):
        self.J = filters.J
        self.L = filters.L
        self.size_x, self.size_y = filters.size, filters.size
        self.filters = filters



    def run(self, input_field):
        """Perform the scattering transform. Computes multiplication and ffts on single tensors."""
        self.I0 = input_field
        I0_k = torch.fft.fft2(self.I0)

        batch_size = input_field.shape[0]


        self.I1 = torch.zeros((batch_size, self.J, self.L, self.size_x, self.size_y), dtype=torch.complex64)
        self.I2 = torch.zeros((batch_size, self.J, self.L, self.J, self.L, self.size_x, self.size_y), dtype=torch.complex64)

        # cheeky dimension tricks to avoid loops and go 10% faster -- does not filter out j2 > j1 though
        product1 = I0_k[:, None, None, ...] * self.filters.filters_k
        self.I1 = torch.fft.ifftn(product1, dim=(-1, -2)).abs()
        I1_k = torch.fft.fftn(self.I1, dim=(-1, -2))


        product2 = I1_k[..., None, None, :, :] * self.filters.filters_k
        self.I2 = torch.fft.ifftn(product2, dim=(-1, -2)).abs()

        return self.calculate_scattering_coefficients()


    def calculate_scattering_coefficients(self):
        self.S0 = torch.mean(self.I0, dim=(-1, -2))
        self.S1 = torch.mean(torch.abs(self.I1), dim=(-1, -2))
        self.S2 = torch.mean(torch.abs(self.I2), dim=(-1, -2))

        self.s0 = self.S0
        self.s1 = torch.mean(self.S1, dim=-1)
        self.s2 = torch.mean(self.S2, dim=(-1, -3))

        return self.s0, self.s1, self.s2

    def rotationally_average_fields(self):
        self.I1_rot_avg = torch.mean(self.I1, dim=-1)
        self.I2_rot_avg = torch.mean(self.I2, dim=(-3, -5))

        return self.I1_rot_avg, self.I2_rot_avg


class ScatteringTransformFast:

    def __init__(self, filters: WaveletsMorlet):
        self.J = filters.J
        self.L = filters.L
        self.size_x, self.size_y = filters.size, filters.size
        self.filters = filters

    def run(self, input_fields, normalised=False, condensed=False):
        """Perform the scattering transform"""
        batch_size = input_fields.shape[0]

        self.I0 = input_fields
        I0_k = torch.fft.fft2(self.I0, dim=(-2, -1))

        self.S0 = torch.mean(torch.abs(self.I0), dim=(-2, -1))
        self.S1 = torch.zeros(size=(batch_size, self.J, self.L))
        self.S2 = torch.full((batch_size, self.J, self.L, self.J, self.L), torch.nan)

        for j1 in range(self.J):
            I0_k_cut, cut_factor = self.cut_high_k_off(I0_k, j=j1)
            product = I0_k_cut[:, None, :, :] * self.filters.filters_cut[j1]
            I1_j1 = torch.fft.ifftn(product, dim=(-2, -1)).abs()
            self.S1[:, j1] = torch.mean(I1_j1, dim=(-2, -1)) * cut_factor

            I1_j1_k = torch.fft.fftn(I1_j1, dim=(-2, -1))


            for j2 in range(self.J):
                if j2 > j1:

                    if j1 >= 1:
                        factor = j2 - j1 + 1
                    else:
                        factor = j2

                    I1_j1_k_cut, cut_factor = self.cut_high_k_off(I1_j1_k, j=factor)
                    product = I1_j1_k_cut[:, :, None, :, :] * self.filters.filters_cut[j2][None, None, :, :, :]
                    I2_j1j2 = torch.fft.ifftn(product, dim=(-2, -1)).abs()

                    self.S2[:, j1, :, j2, :] = torch.mean(I2_j1j2, dim=(-2, -1)) * cut_factor

        self.s0 = self.S0
        self.s1 = torch.mean(self.S1, dim=-1)
        self.s2 = torch.mean(self.S2, dim=(-3, -1))

        print(self.s0.shape)
        print(self.s1.shape)
        print(self.s2.shape)

        if normalised:
            self.s1 = self.s1 / self.s0[:, None]
            self.s2 = self.s2 / self.s1[:, None]

        print('normed')
        print(self.s0.shape)
        print(self.s1.shape)
        print(self.s2.shape)

        if condensed:
            self.s2 = self.s2.flatten(-2, -1)
            # self.s2 = self.s2[~torch.isnan(self.s2)]

            print('condensed')
            print(self.s0.shape)
            print(self.s1.shape)
            print(self.s2.shape)

            return torch.cat([self.s0, self.s1, self.s2.flatten()], dim=-1)

        return self.s0, self.s1, self.s2

    def cut_high_k_off(self, data_k, j=1):
        if j <= 1:
            return data_k, 1

        M = data_k.shape[-2]
        N = data_k.shape[-1]
        dx = int(max(16, min(np.ceil(M / 2 ** j), M // 2)))
        dy = int(max(16, min(np.ceil(N / 2 ** j), N // 2)))

        pre_cut_size = data_k.numel()

        result = torch.cat(
            (torch.cat(
                (data_k[..., :dx, :dy], data_k[..., -dx:, :dy]
                 ), -2),
             torch.cat(
                 (data_k[..., :dx, -dy:], data_k[..., -dx:, -dy:]
                  ), -2)
            ), -1)

        post_cut_size = result.numel()
        cut_factor = post_cut_size / pre_cut_size
        return result, cut_factor


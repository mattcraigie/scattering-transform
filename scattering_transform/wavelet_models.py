import torch
import numpy as np


class WaveletsMorlet(object):
    """
    Standard Morlet Wavelet. Go to, because it has all the properties of a Gabor filter and satisfies the admissibility
    criteria.
    """
    def __init__(self, image_size, J, L, make_filters=True, device='cpu'):
        self.size = image_size
        self.J = J
        self.L = L
        self.x_grid = self._get_x_grid()
        self.filters_x = torch.zeros((self.J, self.L, self.size, self.size), dtype=torch.complex64)
        self.filters_k = torch.zeros((self.J, self.L, self.size, self.size), dtype=torch.complex64)
        self.device = device

        if make_filters:
            self.make_filters()


    def morlet_wavelet_jl(self, j, l):
        """Morlet wavelet constructed directly using j and l with a varied k0 and sigma, as per kymatio"""

        # morlet params for j, l
        x = self.x_grid
        sigma = 0.8 * 2 ** j
        k0_mag = 3 * torch.pi / (4 * 2 ** j)
        s = 4 / self.L
        theta = torch.tensor((int(self.L - self.L / 2 - 1) - l) * torch.pi / self.L)

        a = 1 / sigma
        b = s / sigma
        D = torch.tensor([[a, 0.], [0., b]], dtype=torch.complex64)
        V = self.rotation(theta)
        A = D @ V
        k0 = torch.tensor([[[[k0_mag * torch.cos(theta)], [k0_mag * torch.sin(theta)]]]], dtype=torch.complex64)
        morlet = torch.squeeze(self.morlet(x, A, k0))

        morlet /= 2 * torch.pi * sigma**2 / s

        return morlet

    def make_filters(self):
        for j in range(self.J):
            for l in range(self.L):
                self.filters_x[j, l] = self.morlet_wavelet_jl(j, l)
                self.filters_k[j, l] = torch.fft.fft2(torch.fft.fftshift(self.filters_x[j, l]))
                # self.filters_k[j, l, 0, 0] = 0  # borrowed from yst code, don't know if it actually does anything
        self.apply_filter_cut(self.filters_k)

    def gaussian(self, x):
        return torch.exp(-x.transpose(-2, -1) @ x / 2)

    def sinusoid(self, x, k0):
        return torch.exp(1j * k0.transpose(-2, -1) @ x)

    def admissibility_factor(self, A, k0):
        # this is the same each time, could make more efficient
        return self.gaussian(torch.inverse(A.transpose(-2, -1)) @ k0)

    def morlet(self, x, A, k0):
        return self.gaussian(A @ x) * (self.sinusoid(x, k0) - self.admissibility_factor(A, k0))

    def rotation(self, theta):
        return torch.tensor([[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]],
                            dtype=torch.complex64)

    def _get_x_grid(self):
        pixels = torch.arange(-int(self.size / 2), int(self.size / 2))
        grid_x, grid_y = torch.meshgrid(pixels, pixels)
        grid_xvec = torch.stack([grid_x, grid_y])
        grid_xvec = grid_xvec.swapaxes(0, 1).swapaxes(1, 2)[..., None]
        return grid_xvec.type(torch.complex64)


    def apply_filter_cut(self, filters):
        self.filters_cut = []
        for j in range(self.J):
            self.filters_cut.append(self.cut_high_k_off(filters[j], j))



    def cut_high_k_off(self, data_k, j=1):
        if j <= 1:
            return data_k

        M = data_k.shape[-2]
        N = data_k.shape[-1]
        dx = int(max(16, min(np.ceil(M / 2 ** j), M // 2)))
        dy = int(max(16, min(np.ceil(N / 2 ** j), N // 2)))


        result = torch.cat(
            (torch.cat(
                (data_k[..., :dx, :dy], data_k[..., -dx:, :dy]
                 ), -2),
             torch.cat(
                 (data_k[..., :dx, -dy:], data_k[..., -dx:, -dy:]
                  ), -2)
            ), -1)


        return result.to(self.device)


class CustomFilters:
    """
    Custom filters - note that these are not by default symmetric around pi so we go through the whole 2pi range
    """
    def __init__(self, image_size, J, L, mother_wavelet):
        self.size = image_size
        self.J = J + 1
        self.L = L
        self.x_grid = self._get_x_grid()
        self.filters_x = torch.zeros((self.J, L, self.size, self.size), dtype=torch.complex64)
        self.filters_k = torch.zeros((self.J, L, self.size, self.size), dtype=torch.complex64)
        self.mother_wavelet = mother_wavelet

    def child_wavelet(self, j, l):
        """The child wavelets are real-space dilations and rotations of the mother wavelet"""
        x = self.x_grid
        j = j + 1
        theta = torch.tensor(2 * torch.pi * l / self.L)
        x = self._inv_rotation_operator(x.swapaxes(2, 3), theta).swapaxes(2, 3)  # axis swapping so matrix mult behaves
        return torch.squeeze(self.mother_wavelet(x / 2 ** j) / 2 ** 2 * j)

    def make_filters(self):
        for j in range(self.J):
            for l in range(self.L):
                self.filters_x[j, l] = self.child_wavelet(j, l)
                self.filters_k[j, l] = torch.fft.fft2(torch.fft.fftshift(self.filters_x[j, l]))
                # self.filters_k[j, l, 0, 0] = 0  # borrowed from yst code, don't know if it actually does anything

    def _inv_rotation_operator(self, x, theta):
        """Rotates a vector k by theta"""
        return torch.tensor([[torch.cos(theta), torch.sin(theta)],
                             [-torch.sin(theta), torch.cos(theta)]]) @ x

    def _get_x_grid(self):
        pixels = torch.arange(-int(self.size/2), int(self.size/2))
        grid_x, grid_y = torch.meshgrid(pixels, pixels)
        grid_xvec = torch.stack([grid_x, grid_y])
        grid_xvec = grid_xvec.swapaxes(0, 1).swapaxes(1, 2)[:, :, None, :]
        return grid_xvec.type(torch.float32)



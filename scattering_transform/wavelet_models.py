import torch
import numpy as np


def gaussian(x, x0, cov):
    shifted = x - x0
    return torch.exp(-shifted.transpose(-2, -1) @ cov @ shifted / 2)


def rotation_matrix(theta):
    return torch.tensor([[torch.cos(theta), torch.sin(theta)],
                         [-torch.sin(theta), torch.cos(theta)]])


def gabor(k, k0, sigma):
    return gaussian(k, k0, sigma)


def admissibility_factor(k0, sigma):
    return gaussian(k0, 0, sigma)


def morlet_function(k, k0, sigma):
    return (gabor(k, k0, sigma) - admissibility_factor(k0, sigma) * gaussian(k, 0, sigma)).squeeze(-1).squeeze(-1)


class Wavelet(object):
    def __init__(self, size, num_scales, num_angles):
        self.filter_tensor = torch.zeros((num_scales, num_angles, size, size))
        self.size = size
        self.num_scales = num_scales
        self.num_angles = num_angles
        self.assign_filters()

    def wavelet_function(self, scale, angle):
        raise ValueError("The function 'wavelet_function' must be overriden in the child class")

    def assign_filters(self):
        for scale in range(self.num_scales):
            for angle in range(self.num_angles):
                self.filter_tensor[scale, angle] = self.wavelet_function(scale, angle)

    def to(self, device):
        self.filter_tensor = self.filter_tensor.to(device)


class Morlet(Wavelet):
    def __init__(self, size, num_scales, num_angles):
        super(Morlet, self).__init__(size, num_scales, num_angles)

    def wavelet_function(self, scale, angle):
        """Computes the morlet wavelet. I could make this more efficient by doing the 'extra' trick only for the
        first couple j but this is hardly a bottleneck. """

        extra = 3 if scale < 3 else 1
        kpixels = torch.fft.fftfreq(self.size * extra)

        kx, ky = torch.meshgrid(kpixels, kpixels)
        kpos = torch.stack([kx, ky]).permute(1, 2, 0)[..., None] * extra * np.pi * 2

        sigma = 0.8 * 2 ** scale
        k0_mag = 3 * torch.pi / (4 * 2 ** scale)
        s = 4 / self.num_angles
        theta = torch.tensor((int(self.num_angles - self.num_angles / 2 - 1) - angle) * torch.pi / self.num_angles)

        a = 1 / sigma
        b = s / sigma
        D = torch.tensor([[a, 0.],
                          [0., b]])
        V = rotation_matrix(theta)
        A = D @ V

        k0 = torch.tensor([k0_mag * torch.cos(theta), k0_mag * torch.sin(theta)])[None, None, :, None]

        big_sigma = torch.inverse(A.T @ A)[None, None, :]
        morl = morlet_function(kpos, k0, big_sigma)

        chunked = (torch.tensor_split(b, extra, 0) for b in torch.tensor_split(morl, extra, 1))
        chunked = [portion for tupl in chunked for portion in tupl]
        stacked = torch.stack(chunked).sum(0)

        return stacked


class Learnable(Wavelet, torch.nn.Module):
    def __init__(self, size, num_scales, num_angles):
        Wavelet.__init__(self, size, num_scales, num_angles)
        torch.nn.Module.__init__(self)
        self.filter_tensor = torch.randn_like(self.filter_tensor)
        self.filter_tensor = torch.nn.Parameter(self.filter_tensor)

    def forward(self):
        return

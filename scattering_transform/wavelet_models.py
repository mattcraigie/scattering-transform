import torch
import numpy as np


def gaussian(x):
    return torch.exp(-x.transpose(-2, -1) @ x / 2)


def sinusoid(x, k0):
    return torch.exp(1j * k0.transpose(-2, -1) @ x)


def admissibility_factor(A, k0):
    return gaussian(torch.inverse(A.transpose(-2, -1)) @ k0)


def morlet_function(x, A, k0):
    return gaussian(A @ x) * (sinusoid(x, k0) - admissibility_factor(A, k0))


def rotation(theta):
    return torch.tensor([[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]],
                        dtype=torch.complex64)


class Wavelet(object):
    def __init__(self, size, num_scales, num_angles):
        self.filter_tensor = torch.zeros((num_scales, num_angles, size, size), dtype=torch.complex64)
        self.size = size
        self.num_scales = num_scales
        self.num_angles = num_angles
        self.assign_filters()

    def wavelet_function(self, scale, angle):
        raise ValueError("The function 'wavelet_function' must be overriden in the child class")

    def assign_filters(self):
        for scale in range(self.num_scales):
            for angle in range(self.num_scales):
                self.filter_tensor[scale, angle] = self.wavelet_function(scale, angle)

    def to(self, device):
        self.filter_tensor.to(device)


class Morlet(Wavelet):
    def __init__(self, size, num_scales, num_angles):
        super(Morlet, self).__init__(size, num_scales, num_angles)

    def wavelet_function(self, scale, angle):
        """Morlet wavelet constructed directly using j and l with a varied k0 and sigma, as per kymatio"""
        pixels = torch.arange(- self.size // 2, self.size // 2)
        grid_x, grid_y = torch.meshgrid(pixels, pixels)
        grid_xy = torch.stack([grid_x, grid_y])
        pos = grid_xy.swapaxes(0, 1).swapaxes(1, 2)[..., None].type(torch.complex64)
        sigma = 0.8 * 2 ** scale
        k0_mag = 3 * torch.pi / (4 * 2 ** scale)
        s = 4 / self.num_angles
        theta = torch.tensor((int(self.num_angles - self.num_angles / 2 - 1) - angle) * torch.pi / self.num_angles)

        a = 1 / sigma
        b = s / sigma
        D = torch.tensor([[a, 0.], [0., b]], dtype=torch.complex64)
        V = rotation(theta)
        A = D @ V
        k0 = torch.tensor([[[[k0_mag * torch.cos(theta)], [k0_mag * torch.sin(theta)]]]], dtype=torch.complex64)
        morl = torch.squeeze(morlet_function(pos, A, k0))

        morl /= 2 * torch.pi * sigma**2 / s

        return torch.fft.fft2(torch.fft.fftshift(morl))

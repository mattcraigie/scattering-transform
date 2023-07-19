import torch
import numpy as np
from scipy.special import erf



def create_bank(size, num_scales, num_angles, wavelet_function):
    filter_tensor = torch.zeros(num_scales, num_angles, size, size, dtype=torch.complex64)
    for scale in range(num_scales):
        for angle in range(num_angles):
            filter_tensor[scale, angle] = wavelet_function(size, scale, angle, num_scales, num_angles)
    return filter_tensor



# General functions

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



# Wavelet Functions

def morlet_wavelet(size, scale, angle, num_scales, num_angles):
    """Computes the morlet wavelet. I could make this more efficient by doing the 'extra' trick only for the
    first couple j but this is hardly a bottleneck. """

    extra = 3 if scale < 3 else 1
    kpixels = torch.fft.fftfreq(size * extra)

    kx, ky = torch.meshgrid(kpixels, kpixels)
    kpos = torch.stack([kx, ky]).permute(1, 2, 0)[..., None] * extra * np.pi * 2

    sigma = 0.8 * 2 ** scale
    k0_mag = 3 * torch.pi / (4 * 2 ** scale)
    s = 4 / num_angles
    theta = torch.tensor((int(num_angles - num_angles / 2 - 1) - angle) * torch.pi / num_angles)

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


# def normal(x):
#     return 1 / np.sqrt(2 * np.pi) * np.exp(-np.swapaxes(x, -2, -1) @ x / 2)
#
#
# def cum_dist_func(x):
#     return np.product((1 + erf(x / np.sqrt(2))) / 2, axis=2)[:, :, :, np.newaxis]
#
#
# def skew_normal(x, alpha):
#     return 2 * normal(x) * cum_dist_func(alpha * x)
#
#
# def complex_sinusoid(x, beta):
#     return np.exp(2j * np.pi * np.swapaxes(x, -2, -1) @ beta)[:, :, :, None]


# def skew_wavelet(size, scale, angle, num_scales, num_angles):
#     alpha = np.array([5, 5])[:, np.newaxis]
#     beta = np.array([0.5, 0.5])
#     xx, yy = np.meshgrid(np.linspace(-30, 30, size), np.linspace(-30, 30, size))
#     x = np.array([xx, yy]).swapaxes(0, 2).swapaxes(0, 1)[..., np.newaxis]
#     theta = torch.tensor(2 * (int(num_angles - num_angles / 2 - 1) - angle) * torch.pi / num_angles)
#     x_rotated = x.swapaxes(-2, -1) @ rotation_matrix(theta).numpy()
#     x_scaled_and_rotated = x_rotated.swapaxes(-2, -1) / 2 ** scale
#     wavelet = 2 * skew_normal(x_scaled_and_rotated, alpha) * complex_sinusoid(x_scaled_and_rotated, beta)
#     wavelet = wavelet[:, :, 0, 0]
#     wavelet_k = np.fft.fft2(np.fft.fftshift(wavelet, axes=(0, 1)))
#     return torch.from_numpy(wavelet_k)


# skew wavelet along 1 axis only
def gaussian(x, cov_inv):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-np.swapaxes(x, -2, -1) @ cov_inv @ x / 2)


def phi(x, alpha):
    return ((1 + erf(np.swapaxes(x, -2, -1) @ alpha / np.sqrt(2))) / 2)[:, :, :, None]


def skew_gaussian(x, gaussian_cov, skewness):
    return 2 * gaussian(x, gaussian_cov) * phi(x, skewness)


def complex_sinusoid(x, beta):
    return np.exp(2j * np.pi * np.swapaxes(x, -2, -1) @ beta)[:, :, :, None]


def skew_wavelet(size, scale, angle, num_scales, num_angles):
    xx, yy = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size))
    x = np.array([xx, yy]).swapaxes(0, 2).swapaxes(0, 1)[..., np.newaxis]
    theta = torch.tensor(2 * (int(num_angles - num_angles / 2 - 1) - angle) * torch.pi / num_angles)
    x_rotated = x.swapaxes(-2, -1) @ rotation_matrix(theta).numpy()
    x_scaled_and_rotated = x_rotated.swapaxes(-2, -1) / 2 ** scale

    alpha = np.array([30, 0])
    beta = np.array([3, 0])
    gamma = np.array([[1, 0], [0, 5]]) * 20

    wavelet = 2 * skew_gaussian(x_scaled_and_rotated, gamma, alpha) * complex_sinusoid(x_scaled_and_rotated, beta)
    wavelet = wavelet[:, :, 0, 0]
    wavelet_k = np.fft.fft2(np.fft.fftshift(wavelet, axes=(0, 1)))
    return torch.from_numpy(wavelet_k)







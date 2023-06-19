import torch
from torch.fft import fft2
import numpy as np


class PowerSpectrum(torch.nn.Module):
    def __init__(self, size, num_bins):
        super(PowerSpectrum, self).__init__()
        """A power spectrum claculator. Measures the power spectrum of a 2D field and bins into rotationally averaged, 
        log-spaced bins."""

        self.size = size
        self.num_bins = num_bins
        self.bin_masks, self.bin_mask_sums = self.make_bin_masks(size, num_bins)

    def make_bin_masks(self, size, num_bins):
        """Makes a set of masks for binning the power spectrum. The bins are log-spaced, and the first bin is centered
        at 0. The last bin is centered at the Nyquist frequency. The bins are rotationally averaged."""

        # Make a grid of k values
        kpixels = torch.fft.fftfreq(size) * size
        kx, ky = torch.meshgrid(kpixels, kpixels)
        k = torch.sqrt(kx ** 2 + ky ** 2)

        # Make a set of masks for binning the power spectrum
        bin_edges = torch.logspace(0, np.log2(size / 2), num_bins + 1, base=2)
        bin_masks = torch.zeros(num_bins, size, size)
        for i in range(num_bins):
            bin_masks[i] = (k > bin_edges[i]) * (k <= bin_edges[i + 1])
        bin_mask_sums = torch.sum(bin_masks, dim=(-1, -2))
        return bin_masks, bin_mask_sums

    def forward(self, x):
        # x has input shape (batch, sub-batch, ..., size, size)
        # output has shape (batch, sub-batch, ..., num_bins)
        batch_dims = len(x.shape[:-2])

        # Compute the power spectrum
        x = fft2(x, norm='ortho').abs() ** 2

        # Bin the power spectrum
        x = x[..., None, :, :]
        x = x.repeat((1, 1, self.num_bins, 1, 1))
        x = torch.sum(x * self.bin_masks[(None,)*batch_dims + (Ellipsis,)], dim=(-2, -1)) / self.bin_mask_sums
        return x

    def to(self, device):
        super(PowerSpectrum, self).to(device)
        self.bin_masks = self.bin_masks.to(device)
        self.bin_mask_sums = self.bin_mask_sums.to(device)
        return self


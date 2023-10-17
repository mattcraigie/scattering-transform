import torch


def scattering_operation(input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
    """
    Convolves and takes the absolute value of two  2D input fields
    :param input_a: the first field in Fourier space (usually a physical field)
    :param input_b: the second field in Fourier space (usually a wavelet filter)
    :return: the output scattering field
    """
    return torch.fft.ifft2(input_a * input_b).abs()


def scattering_operation_3d(input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
    """
    Convolves and takes the absolute value of two 3D input fields
    :param input_a: the first field in Fourier space (usually a physical field)
    :param input_b: the second field in Fourier space (usually a wavelet filter)
    :return: the output scattering field
    """
    return torch.fft.ifftn(input_a * input_b, dim=(-3, -2, -1)).abs()


def clip_fourier_field(field: torch.Tensor, final_size: int) -> torch.Tensor:
    """
    Performs a high frequency clip on a 2D field in Fourier space for a filter scale J.
    :param field: The input field.
    :param final_size: The final size
    :return: the field after clipping
    """
    assert final_size <= field.shape[-1], "Final size must be smaller than or same as the input field size"
    cl = final_size // 2
    result = torch.cat((torch.cat((field[..., :cl, :cl], field[..., -cl:, :cl]), -2),
                        torch.cat((field[..., :cl, -cl:], field[..., -cl:, -cl:]), -2)
                        ), -1)
    return result


def clip_fourier_field_3d(field: torch.Tensor, final_size: int) -> torch.Tensor:
    """
    Performs a high frequency clip on a 2D field in Fourier space for a filter scale J.
    :param field: The input field.
    :param final_size: The final size
    :return: the field after clipping
    """
    assert final_size <= field.shape[-1], "Final size must be smaller than or same as the input field size"
    cl = final_size // 2
    result = torch.cat([
                torch.cat((torch.cat((field[..., :cl, :cl, :cl], field[..., -cl:, :cl, :cl]), -3),
                        torch.cat((field[..., :cl, -cl:, :cl], field[..., -cl:, -cl:, :cl]), -3)
                        ), -2),

                torch.cat((torch.cat((field[..., :cl, :cl, -cl:], field[..., -cl:, :cl, -cl:]), -3),
                        torch.cat((field[..., :cl, -cl:, -cl:], field[..., -cl:, -cl:, -cl:]), -3)
                        ), -2)
                ], -1)

    return result


def dyadic_clip_sizes(scale: int, start_size: int):
    """
    The default clip size, halving with each increase in scale.
    :param scale: The wavelet scale factor (often denoted j)
    :param start_size: The size of the initial field
    :return:
    """
    return int(start_size * 2 ** (-scale))


def morlet_clip_sizes(scale: int, start_size: int):
    """
    The clip size that works with Morlet wavelets. This comes from the unusual specification, borrowed from kymatio.
    :param scale: The wavelet scale factor (often denoted j)
    :param start_size: The size of the initial field
    :return:
    """
    return min(max(int(start_size * 2 ** (-scale + 1)), 32), start_size)
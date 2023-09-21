import torch


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
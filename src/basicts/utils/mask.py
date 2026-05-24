import numpy as np
import torch


def null_val_mask(data: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Create a mask for the input data.

    Args:
        data (torch.Tensor): Input data.
        mask_ratio (float): Ratio of the data to be masked.

    Returns:
        torch.Tensor: Mask for the input data.
    """

    if np.isnan(null_val):
        return ~torch.isnan(data)

    eps = 5e-5
    if null_val == 0.0:
        return data != 0.0
    return torch.abs(data - float(null_val)) > eps

def reconstruction_mask(data: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    Create a mask for the input data of a masked reconstruction task.

    Args:
        data (torch.Tensor): Input data.
        mask_ratio (float): Ratio of the data to be masked.

    Returns:
        torch.Tensor: Mask for the input data.
    """

    return torch.rand_like(data) > mask_ratio # 0 for masked, 1 for remained

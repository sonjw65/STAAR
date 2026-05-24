import numpy as np
import torch


def masked_smape(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    targets_mask: torch.Tensor = None,
    null_val: float = np.nan,
) -> torch.Tensor:
    """
    Calculate the Masked Symmetric Mean Absolute Percentage Error (SMAPE) between predicted and target values,
    ignoring entries that are either zero or match the specified null value in the target tensor.

    This function is particularly useful for time series or regression tasks where the target values may 
    contain zeros or missing values, which could otherwise distort the error calculation. The function 
    applies a mask to ensure these entries do not affect the resulting MAPE.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        targets (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Defaults to `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute percentage error.

    Details:
        - The function creates two masks:
          1. `zero_mask`: This mask excludes entries in the `target` tensor that are close to zero, 
             since division by zero or near-zero values would result in extremely large or undefined errors.
          2. `null_mask`: This mask excludes entries in the `target` tensor that match the specified `null_val`. 
             If `null_val` is `np.nan`, the mask will exclude `NaN` values using `torch.isnan`.
        
        - The final mask is the intersection of `zero_mask` and `null_mask`, ensuring that only valid, non-zero,
          and non-null values contribute to the MAPE calculation.
    """

    # mask to exclude zero values in the target
    zero_mask = ~torch.isclose(targets, torch.tensor(0.0, device=targets.device), atol=5e-5)

    # Prefer the precomputed task mask when available.
    if targets_mask is not None:
        null_mask = targets_mask.bool()
    elif np.isnan(null_val):
        null_mask = ~torch.isnan(targets)
    else:
        eps = 5e-5
        null_mask = ~torch.isclose(targets, torch.tensor(null_val, device=targets.device), atol=eps)

    # combine zero and null masks
    mask = (zero_mask & null_mask).float()

    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = torch.abs(prediction - targets) / ((prediction.abs() + targets.abs()) / 2)
    loss *= mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)

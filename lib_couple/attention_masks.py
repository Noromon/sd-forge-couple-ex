"""
Credit: laksjdjf
https://github.com/laksjdjf/cgem156-ComfyUI/blob/main/scripts/attention_couple/node.py
"""

import math

import torch
from torch.nn.functional import interpolate


def repeat_div(value: int, iterations: int) -> int:
    for _ in range(iterations):
        value = math.ceil(value / 2)

    return value


def get_mask(mask: torch.Tensor, batch_size: int, num_tokens: int, shape: tuple[int]):
    """
    Credit: hako-mikan
    https://github.com/hako-mikan/sd-webui-regional-prompter/blob/main/scripts/attention.py

    Issue Found/Fixed by. arcusmaximus & www
    https://github.com/arcusmaximus/sd-forge-couple/tree/draggable-box-ui
    """

    width, height = shape[3], shape[2]

    scale = math.ceil(math.log2(math.sqrt(height * width / num_tokens)))
    size = (repeat_div(height, scale), repeat_div(width, scale))

    num_conds = mask.shape[0]
    mask_downsample = interpolate(mask, size=size, mode="nearest")
    mask_downsample = mask_downsample.view(num_conds, num_tokens, 1)

    return mask_downsample.repeat_interleave(batch_size, dim=0)


def get_dit_mask(mask: torch.Tensor, seq_len: int, w: int, h: int, patch_size: int = 2):
    """Dynamically resizes and flattens the 2D mask to match the DiT's sequence length"""

    num_conds = mask.shape[0]

    h_p = h // (8 * patch_size)
    w_p = w // (8 * patch_size)
    t_p = max(seq_len // (h_p * w_p), 1)

    mask_for_interp = mask.view(num_conds, 1, mask.shape[-2], mask.shape[-1])

    mask_resized = interpolate(
        mask_for_interp, size=(h_p, w_p), mode="bilinear", align_corners=False
    )

    mask_flattened = mask_resized.view(num_conds, 1, h_p * w_p)
    mask_flattened = mask_flattened.repeat(1, t_p, 1)

    return mask_flattened.view(num_conds, 1, seq_len, 1)


def sharpen_mask(mask: torch.Tensor, mode: str, temperature: float) -> torch.Tensor:
    """Sharpen the mask to increase regional separation.

    Args:
        mask: mask tensor of shape [num_conds, ...] (already normalized so sum=1 along dim 0)
        mode: "Soft" (no change), "Hard" (argmax winner-take-all), "Temperature" (power sharpening)
        temperature: sharpening temperature, lower = sharper. Only used in "Temperature" mode.

    Returns:
        Sharpened mask with the same shape.
    """
    if mode == "Soft" or temperature >= 1.0:
        return mask

    if mode == "Hard":
        # Winner-take-all: each spatial position belongs entirely to one region
        hard_idx = mask.argmax(dim=0, keepdim=True)
        result = torch.zeros_like(mask)
        result.scatter_(0, hard_idx, 1.0)
        return result

    # Temperature mode: sharpen via power law
    mask = mask.clamp(min=1e-6) ** (1.0 / temperature)
    mask = mask / mask.sum(dim=0, keepdim=True)
    return mask


def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def lcm_for_list(numbers: list[int]) -> int:
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm

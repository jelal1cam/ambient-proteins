import math
import torch

# Cache for arange tensors by (D, device) to avoid repeated allocations
_arange_cache = {}

@torch.compiler.disable  # Prevent tracing into cache logic
def _get_arange(D, device):
    """Get cached arange tensor for sinusoidal encoding."""
    key = (D, device)
    if key not in _arange_cache:
        _arange_cache[key] = torch.arange(1, D+1, device=device)
    return _arange_cache[key]


def sinusoidal_encoding(v, N, D):
    # v: [*]

    # [D] - use cached tensor to avoid repeated allocations
    k = _get_arange(D, v.device)

    # [*, D]
    sin_div_term = N ** (2 * k / D)
    sin_div_term = sin_div_term.view(*((1, ) * len(v.shape) + (len(sin_div_term), )))
    sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)

    # [*, D]
    cos_div_term = N ** (2 * (k - 1) / D)
    cos_div_term = cos_div_term.view(*((1, ) * len(v.shape) + (len(cos_div_term), )))
    cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)

    # [*, D] - zeros_like already inherits device, no .to() needed
    enc = torch.zeros_like(sin_enc)
    enc[..., 0::2] = cos_enc[..., 0::2]
    enc[..., 1::2] = sin_enc[..., 1::2]

    return enc
import typing as tp
import omegaconf
import torch
import math
from torch import nn
from functools import partial
from torch.nn.utils.rnn import pad_sequence

# Removed imports: flashy, flashy.distrib

def dict_from_config(cfg: tp.Any) -> tp.Any:
    """
    Convert an omegaconf config to a python dict, resolving variable interpolation.
    """
    if isinstance(cfg, omegaconf.DictConfig):
        return omegaconf.OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, omegaconf.ListConfig):
        return omegaconf.OmegaConf.to_container(cfg, resolve=True)
    return cfg

def length_to_mask(lengths: torch.Tensor, max_len: tp.Optional[int] = None) -> torch.Tensor:
    """Utility function to convert a tensor of sequence lengths to a mask (useful when working on padded sequences).
    For example: [3, 5] => [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

    Args:
        lengths (torch.Tensor): tensor with lengths
        max_len (int): can set the max length manually. Defaults to None.
    Returns:
        torch.Tensor: mask with 0s where there is pad tokens else 1s
    """
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(final_length, 1)  # if all seqs are of len zero we don't want a zero-size tensor
    return torch.arange(final_length)[None, :].to(lengths.device) < lengths[:, None]

def collate(tensors: tp.List[torch.Tensor], dim: int = 0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Get a list of tensors and collate them to a single tensor. according to the following logic:
    - `dim` specifies the time dimension which will be stacked and padded.
    - The output will contain 1 new dimension (dimension index 0) which will be the size of
    of the original list.

    Args:
        tensors (tp.List[torch.Tensor]): List of tensors to collate.
        dim (int): Dimension which will be stacked and padded.
    Returns:
        tp.Tuple[torch.Tensor, torch.Tensor]:
            torch.Tensor: Stacked and padded tensor. The output will contain 1 new dimension
                (dimension index 0) which will be the size of the original list.
            torch.Tensor: Tensor containing length of original tensor sizes (without padding).
    """
    tensors = [x.transpose(0, dim) for x in tensors]
    lens = torch.LongTensor([len(x) for x in tensors])
    padded_tensors = pad_sequence(tensors)
    padded_tensors = padded_tensors.transpose(0, 1)
    padded_tensors = padded_tensors.transpose(1, dim + 1)
    return padded_tensors, lens

def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """Create normalization module for transformer encoder layer."""
    if norm_type == 'layer_norm':
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    """LM layer initialization."""
    std = 1 / math.sqrt(input_dim)
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")

def init_layer(m: nn.Module,
               method: str,
               init_depth: tp.Optional[int] = None,
               zero_bias_init: bool = False):
    """Wrapper around ``get_init_fn`` for proper initialization of LM modules."""
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)

def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token

def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output

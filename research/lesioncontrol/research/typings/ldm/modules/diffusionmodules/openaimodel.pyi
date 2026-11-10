from collections.abc import Callable, Sequence

from torch import Tensor, nn

def checkpoint(
    func: Callable[..., Tensor],
    inputs: Sequence[Tensor | None],
    params: Sequence[Tensor],
    flag: bool,
) -> Tensor: ...

class UNetModel(nn.Module):
    def __init__(
        self,
        image_size: int | Sequence[int],
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Sequence[int],
        *,
        channel_mult: Sequence[int] = ...,
        num_head_channels: int = ...,
        use_checkpoint: bool = ...,
        use_fp16: bool = ...,
        use_spatial_transformer: bool = ...,
        skip_context: bool = ...,
        context_dim: int | None = ...,
        dims: int = ...,
    ) -> None: ...

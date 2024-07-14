from ...config import CommonConfig
from ...modeling_utils import PaddingFreeAttention
from .base import _BaseAttention_TP


class PaddingFreeAttention_TP(_BaseAttention_TP, PaddingFreeAttention):
    def __init__(
        self,
        config: CommonConfig,
        causal: bool,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        _BaseAttention_TP.__init__(
            self,
            config=config,
            causal=causal,
            layer_idx=layer_idx,
            use_padding_free_transformer=True,
            sequence_parallel=sequence_parallel,
        )

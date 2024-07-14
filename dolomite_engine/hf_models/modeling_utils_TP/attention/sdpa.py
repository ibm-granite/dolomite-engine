from ...config import CommonConfig
from ...modeling_utils import SDPA
from .base import Attention_TP


class SDPA_TP(Attention_TP, SDPA):
    def __init__(
        self,
        config: CommonConfig,
        causal: bool,
        layer_idx: int | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        Attention_TP.__init__(
            self,
            config=config,
            causal=causal,
            layer_idx=layer_idx,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

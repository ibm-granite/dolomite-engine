from ...mixins import CausalLMModelMixin
from .base import GPTDolomiteModel, GPTDolomitePreTrainedModel


class GPTDolomiteForCausalLM(GPTDolomitePreTrainedModel, CausalLMModelMixin):
    _tied_weights_keys = ["lm_head.weight"]
    base_model_class = GPTDolomiteModel

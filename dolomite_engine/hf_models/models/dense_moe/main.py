from typing import List, Tuple, Union

import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from ..gpt_megatron import GPTMegatronForCausalLM
from .base import DenseMoEModel, DenseMoEPreTrainedModel
from .config import DenseMoEConfig


class DenseMoEForCausalLM(DenseMoEPreTrainedModel, GPTMegatronForCausalLM):
    def __init__(self, config: DenseMoEConfig, **kwargs) -> None:
        DenseMoEPreTrainedModel.__init__(self, config, **kwargs)

        self.transformer = DenseMoEModel(config, **kwargs)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]] = None,
        past_key_values: DynamicCache = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: Union[torch.Tensor, List[List[int]]] = None,
        position_ids: Union[torch.Tensor, List[List[int]]] = None,
        inputs_embeds: Union[torch.Tensor, List[List[float]]] = None,
        labels: Union[torch.Tensor, List[List[int]]] = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, position_ids, token_type_ids, labels, cu_seqlens, max_seqlen = self.prepare_inputs_for_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            labels=labels,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # ==========================================================================================
        # padding_free:
        #     input_ids -> (total_q)
        #     attention_mask -> None
        #     position_ids -> (total_q)
        # else:
        #     input_ids -> (batch_size, query_length)
        #     attention_mask -> None or (batch_size, key_length)
        #     position_ids -> None or (batch_size, key_length)
        # ==========================================================================================

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        loss = self.get_autoregressive_language_modeling_loss(lm_logits, labels, cu_seqlens)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

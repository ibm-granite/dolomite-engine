import torch
from tqdm import trange

from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import is_glu
from ..gpt_dolomite import GPTDolomiteConfig


def unshard(
    config: GPTDolomiteConfig, tensor_parallel_state_dicts: list[dict], tensor_parallel_embeddings: bool
) -> dict:
    attention_head_type = AttentionHeadType(config.attention_head_type)
    position_embedding_type = PositionEmbeddingType(config.position_embedding_type)

    # word embeddings
    output_state_dict = _get_embeddings_or_lm_head(
        tensor_parallel_state_dicts,
        tensor_parallel_embeddings=tensor_parallel_embeddings,
        prefix="transformer.wte.weight",
    )

    # positional embeddings if using learned positional embeddings
    if position_embedding_type == PositionEmbeddingType.learned_absolute:
        output_state_dict.update(
            _get_embeddings_or_lm_head(
                tensor_parallel_state_dicts,
                tensor_parallel_embeddings=tensor_parallel_embeddings,
                prefix="transformer.wpe.weight",
            )
        )

    # layers
    for layer_idx in trange(config.n_layer):
        # first layernorm
        output_state_dict.update(
            _get_layernorm(
                tensor_parallel_state_dicts,
                prefix=f"transformer.h.{layer_idx}.ln_1.",
                normalization_function=config.normalization_function,
            )
        )

        # attention
        output_state_dict.update(
            _get_attention(
                tensor_parallel_state_dicts,
                attention_head_type=attention_head_type,
                add_bias=config.add_bias,
                prefix=f"transformer.h.{layer_idx}.attn.",
            )
        )

        # second layernorm
        output_state_dict.update(
            _get_layernorm(
                tensor_parallel_state_dicts,
                prefix=f"transformer.h.{layer_idx}.ln_2.",
                normalization_function=config.normalization_function,
            )
        )

        # mlp
        output_state_dict.update(
            _get_mlp(
                tensor_parallel_state_dicts,
                is_glu=is_glu(config.activation_function),
                add_bias=config.add_bias,
                prefix=f"transformer.h.{layer_idx}.mlp.",
            )
        )

    # final layernorm
    output_state_dict.update(
        _get_layernorm(
            tensor_parallel_state_dicts,
            prefix=f"transformer.ln_f.",
            normalization_function=config.normalization_function,
        )
    )

    if not config.tie_word_embeddings:
        output_state_dict.update(
            _get_embeddings_or_lm_head(
                tensor_parallel_state_dicts,
                tensor_parallel_embeddings=tensor_parallel_embeddings,
                prefix="lm_head.weight",
            )
        )

    return output_state_dict


def _get_embeddings_or_lm_head(
    tensor_parallel_state_dicts: list[dict], tensor_parallel_embeddings: bool, prefix: str
) -> dict:
    output = (
        _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts, key=prefix, dim=0)
        if tensor_parallel_embeddings
        else _get_once_from_state_dicts_with_check(tensor_parallel_state_dicts, key=prefix)
    )
    return {prefix: output}


def _get_layernorm(tensor_parallel_state_dicts: list[dict], prefix: str, normalization_function: str) -> dict:
    output = {
        prefix + "weight": _get_once_from_state_dicts_with_check(tensor_parallel_state_dicts, key=prefix + "weight")
    }
    if normalization_function == "layernorm":
        output[prefix + "bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "bias"
        )
    return output


def _get_attention(
    tensor_parallel_state_dicts: list[dict], attention_head_type: AttentionHeadType, add_bias: bool, prefix: str
) -> dict:
    output = {
        prefix
        + "c_proj.weight": _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_proj.weight", dim=1
        )
    }
    if add_bias:
        output[prefix + "c_proj.bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "c_proj.bias"
        )

    if attention_head_type == AttentionHeadType.mha:
        pass
    elif attention_head_type == AttentionHeadType.gqa:
        pass
    elif attention_head_type == AttentionHeadType.mqa:
        q_weight = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_attn.q_attn.weight", dim=0
        )
        kv_weight = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "c_attn.kv_attn.weight"
        )
        output[prefix + "c_attn.weight"] = torch.cat([q_weight, kv_weight])
        if add_bias:
            q_bias = _concatenate_tensors_from_state_dicts(
                tensor_parallel_state_dicts, key=prefix + "c_attn.q_attn.bias", dim=0
            )
            kv_bias = _get_once_from_state_dicts_with_check(
                tensor_parallel_state_dicts, key=prefix + "c_attn.kv_attn.bias"
            )
            output[prefix + "c_attn.bias"] = torch.cat([q_bias, kv_bias])
    else:
        raise ValueError(f"unexpected attention_head_type ({attention_head_type})")

    return output


def _get_mlp(tensor_parallel_state_dicts: list[dict], is_glu: bool, add_bias: bool, prefix: str) -> dict:
    output = {
        prefix
        + "c_proj.weight": _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_proj.weight", dim=1
        )
    }
    if add_bias:
        output[prefix + "c_proj.bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "c_proj.bias"
        )

    if is_glu:
        weights = [state_dict[prefix + "c_fc.weight"].chunk(2) for state_dict in tensor_parallel_state_dicts]
        weights = (torch.cat([w[0] for w in weights]), torch.cat([w[1] for w in weights]))
        output[prefix + "c_fc.weight"] = torch.cat(weights)
        if add_bias:
            bias = [state_dict[prefix + "c_fc.bias"].chunk(2) for state_dict in tensor_parallel_state_dicts]
            bias = (torch.cat([b[0] for b in bias]), torch.cat([b[1] for b in bias]))
            output[prefix + "c_fc.bias"] = torch.cat(bias)
    else:
        output[prefix + "c_fc.weight"] = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_fc.weight", dim=0
        )
        if add_bias:
            output[prefix + "c_fc.bias"] = _concatenate_tensors_from_state_dicts(
                tensor_parallel_state_dicts, key=prefix + "c_fc.bias", dim=0
            )

    return output


def _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts: list[dict], key: str, dim: int) -> torch.Tensor:
    tensor_list = [state_dict[key] for state_dict in tensor_parallel_state_dicts]
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor


def _get_once_from_state_dicts_with_check(
    tensor_parallel_state_dicts: list[dict], key: str, check_equal: bool = True
) -> torch.Tensor:
    output: torch.Tensor = tensor_parallel_state_dicts[0][key]
    if check_equal:
        for state_dict in tensor_parallel_state_dicts[1:]:
            assert output.equal(state_dict[key])
    return output

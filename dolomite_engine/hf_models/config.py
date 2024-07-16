from transformers import PretrainedConfig

from .enums import AttentionHeadType, InitMethod, PositionEmbeddingType


class CommonConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        num_key_value_heads: int | None = None,
        n_inner: int | None = None,
        activation_function: str = "gelu_pytorch_tanh",
        attention_head_type: str = "mqa",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        normalization_function: str = "layernorm",
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        attention_multiplier: float | None = None,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        attention_softmax_in_fp32: bool = True,
        add_bias: bool = True,
        position_embedding_type: str = "learned_absolute",
        rope_theta: int = 10000,
        rope_scaling: dict | None = None,
        m_emb: float | None = None,
        m_width: float | None = None,
        m_residual: float | None = None,
        init_method: str = "normal",
        upcast_logits_for_loss: bool = False,
        **kwargs,
    ) -> None:
        """A common config class that is inherited by the config classes of almost all models.

        Args:
            vocab_size (int, optional): word embedding matrix size. Defaults to 50257.
            n_positions (int, optional): position embedding matrix size. Defaults to 1024.
            n_embd (int, optional): embedding dimension. Defaults to 768.
            n_layer (int, optional): number of layers. Defaults to 12.
            n_head (int, optional): number of query heads. Defaults to 12.
            num_key_value_heads (int | None, optional): number of key value heads. Defaults to None.
            n_inner (int | None, optional): FFN inner dimension. Defaults to None.
            activation_function (str, optional): activation function. Defaults to "gelu_pytorch_tanh".
            attention_head_type (str, optional): attention head type, can be "mha", "mqa" or "gqa". Defaults to "mqa".
            resid_pdrop (float, optional): residual dropout. Defaults to 0.1.
            embd_pdrop (float, optional): embedding dropout. Defaults to 0.1.
            attn_pdrop (float, optional): attention dropout. Defaults to 0.1.
            normalization_function (str, optional): normalization function. Defaults to "layernorm".
            layer_norm_epsilon (float, optional): tolerance for normalization function. Defaults to 1e-5.
            initializer_range (float, optional): hyperparam to use for initialization, can be standard deviation etc. Defaults to 0.02.
            scale_attn_weights (bool, optional): whether to scale attention QK^T. Defaults to True.
            attention_multiplier (float | None, optional): attention multiplier for scaling. None means `1 / sqrt(head_dim)`. Defaults to None.
            use_cache (bool, optional): whether to use cache during generation. Defaults to True.
            bos_token_id (int, optional): bos token id. Defaults to 50256.
            eos_token_id (int, optional): eos token id. Defaults to 50256.
            pad_token_id (int, optional): pad token id. Defaults to 50256.
            attention_softmax_in_fp32 (bool, optional): whether to compute attention softmax in fp32 for eager attention implementation. Defaults to True.
            scale_attention_softmax_in_fp32 (bool, optional): . Defaults to True.
            add_bias (bool, optional): _description_. Defaults to True.
            position_embedding_type (str, optional): _description_. Defaults to "learned_absolute".
            rope_theta (int, optional): _description_. Defaults to 10000.
            rope_scaling (dict | None, optional): _description_. Defaults to None.
            m_emb (float | None, optional): _description_. Defaults to None.
            m_width (float | None, optional): _description_. Defaults to None.
            m_residual (float | None, optional): _description_. Defaults to None.
            init_method (str, optional): _description_. Defaults to "normal".
            upcast_logits_for_loss (bool, optional): _description_. Defaults to False.
        """

        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_key_value_heads = num_key_value_heads
        self.n_inner = 4 * n_embd if n_inner is None else n_inner
        self.activation_function = activation_function
        self.attention_head_type = attention_head_type
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.normalization_function = normalization_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.attention_multiplier = attention_multiplier
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.position_embedding_type = position_embedding_type
        self.add_bias = add_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.m_emb = m_emb
        self.m_width = m_width
        self.m_residual = m_residual
        self.init_method = init_method
        self.upcast_logits_for_loss = upcast_logits_for_loss

        if self.attention_multiplier is not None:
            assert self.scale_attn_weights

        # check if enums are valid
        init_method = InitMethod(init_method)
        attention_head_type = AttentionHeadType(attention_head_type)
        position_embedding_type = PositionEmbeddingType(position_embedding_type)

        # for compatibility with some features
        self.multi_query = attention_head_type == AttentionHeadType.mqa

        if attention_head_type == AttentionHeadType.mha:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = self.n_head

            assert (
                self.n_head == self.num_key_value_heads
            ), "MultiHeadAttention should have same number of heads for query, keys and values"
        elif attention_head_type == AttentionHeadType.mqa:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = 1

            assert self.num_key_value_heads == 1, "MultiQueryAttention should have 1 head for keys and values"
        elif attention_head_type == AttentionHeadType.gqa:
            assert (
                self.num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            assert (
                self.n_head % self.num_key_value_heads == 0
            ), "GroupedQueryAttention should have more than 1 head for keys and values"

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)

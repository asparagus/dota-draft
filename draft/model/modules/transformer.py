import dataclasses

import torch
from torch import nn


def scaled_dot_product_attention(
        v: torch.Tensor,  # (batch, n_token, dim)
        k: torch.Tensor,  # (batch, n_token, dim)
        q: torch.Tensor,  # (batch, n_token, dim)
        scale: float,
    ) -> torch.Tensor:
    # (batch, n_token, n_token)
    dot_product = k @ q.transpose(-2, -1)
    scaled_dot_product = dot_product * scale
    attention = torch.softmax(scaled_dot_product, dim=-1)

    # (batch, n_token, dim)
    attended_values = attention @ v
    return attended_values


@dataclasses.dataclass
class MultiHeadAttentionConfig:
    n_heads: int
    d_model: int


class MultiHeadAttention(nn.Module):
    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.config = config
        self.V = nn.Linear(in_features=config.d_model, out_features=config.d_model * config.n_heads)
        self.K = nn.Linear(in_features=config.d_model, out_features=config.d_model * config.n_heads)
        self.Q = nn.Linear(in_features=config.d_model, out_features=config.d_model * config.n_heads)
        self.projection = nn.Linear(in_features=config.d_model * config.n_heads, out_features=config.d_model)

    def forward(
            self,
            value_input: torch.Tensor,
            key_input: torch.Tensor,
            query_input: torch.Tensor) -> torch.Tensor:
        # (batch, n_token, dim)
        vs = torch.split(self.V(value_input), split_size_or_sections=self.config.d_model, dim=-1)
        ks = torch.split(self.K(key_input), split_size_or_sections=self.config.d_model, dim=-1)
        qs = torch.split(self.Q(query_input), split_size_or_sections=self.config.d_model, dim=-1)

        # [(batch, n_token, dim)]
        multihead_output = [
            scaled_dot_product_attention(v, k, q, scale=self.scale)
            for v, k, q in zip(vs, ks, qs)
        ]
        # (batch, n_token, dim * n_heads)
        concat = torch.concat(multihead_output, dim=-1)
        # (batch, n_token, dim)
        out = self.projection(concat)
        return out


@dataclasses.dataclass
class TransformerLayerConfig:
    n_heads: int
    d_model: int
    context_length: int


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        attention_config = MultiHeadAttentionConfig(n_heads=config.n_heads, d_model=config.d_model)
        layer_shape = [config.context_length, config.d_model]
        self.config = config

        self.self_attention = MultiHeadAttention(config=attention_config)
        self.self_attention_layer_norm = nn.LayerNorm(normalized_shape=layer_shape)

        self.linear_projection = nn.Sequential(
            nn.Linear(in_features=config.d_model, out_features=config.d_model),
            nn.ReLU(),
            nn.Linear(in_features=config.d_model, out_features=config.d_model),
        )
        self.linear_layer_norm = nn.LayerNorm(normalized_shape=layer_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, n_token, dim)
        attended_value = self.self_attention(
            value_input=x,
            key_input=x,
            query_input=x,
        )
        x = self.self_attention_layer_norm(x + attended_value)
        x = self.linear_layer_norm(x + self.linear_projection(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        attention_config = MultiHeadAttentionConfig(n_heads=config.n_heads, d_model=config.d_model)
        layer_shape = [config.context_length, config.d_model]
        self.config = config

        self.self_attention = MultiHeadAttention(config=attention_config)
        self.self_attention_layer_norm = nn.LayerNorm(normalized_shape=layer_shape)

        self.cross_attention = MultiHeadAttention(config=attention_config)
        self.cross_attention_layer_norm = nn.LayerNorm(normalized_shape=layer_shape)

        self.linear_projection = nn.Sequential(
            nn.Linear(in_features=config.d_model, out_features=config.d_model),
            nn.ReLU(),
            nn.Linear(in_features=config.d_model, out_features=config.d_model),
        )
        self.linear_layer_norm = nn.LayerNorm(normalized_shape=layer_shape)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        # (batch, n_token, dim)
        self_attention_output = self.self_attention(
            value_input=x,
            key_input=x,
            query_input=x,
        )
        x = self.self_attention_layer_norm(x + self_attention_output)

        # (batch, n_token, dim)
        cross_attention_output = self.cross_attention(
            value_input=encoder_output,
            key_input=encoder_output,
            query_input=x,
        )
        x = self.cross_attention_layer_norm(x + cross_attention_output)

        # (batch, n_token, dim)
        x = self.linear_layer_norm(x + self.linear_projection(x))
        return x


@dataclasses.dataclass
class TransformerConfig:
    n_layers: int
    n_heads: int
    d_model: int
    context_length: int


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__(self)
        layer_config = TransformerLayerConfig(
            n_heads=config.n_heads,
            d_model=config.d_model,
            context_length=config.context_length,
        )
        self.encoder = nn.Sequential(*[EncoderLayer(config=layer_config) for _ in range(config.n_layers)])
        self.decoder = nn.Sequential(*[DecoderLayer(config=layer_config) for _ in range(config.n_layers)])
        self.linear = nn.Sequential(
            nn.Linear(in_features=config.d_model, out_features=config.d_model),
            nn.ReLU(),
            nn.Linear(in_features=config.d_model, out_features=config.d_model),
        )

    def forward(
            self,
            encoder_input: torch.Tensor,
            decoder_input: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output=encoder_output)
        linear_output = self.linear(decoder_output)
        return linear_output

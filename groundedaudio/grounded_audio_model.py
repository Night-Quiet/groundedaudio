# coding=utf-8
# Copyright 2024 IDEA Research and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Grounded Audio model."""

import copy
import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    is_scipy_available,
    is_timm_available,
    requires_backends,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import meshgrid
from transformers.utils import is_accelerate_available, logging
from transformers.utils.backbone_utils import load_backbone
from transformers import AutoModel
from .sensevoice_model import SenseVoiceSmall
from .configuration_grounded_audio import GroundedAudioConfig

if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

if is_timm_available():
    from timm import create_model


logger = logging.get_logger(__name__)

MultiScaleDeformableAttention = None


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.MultiScaleDeformableAttentionFunction
class MultiScaleDeformableAttentionFunction(Function):
    @staticmethod
    def forward(
        context,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        context.im2col_step = im2col_step
        output = MultiScaleDeformableAttention.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            context.im2col_step,
        )
        context.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(context, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = context.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MultiScaleDeformableAttention.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            context.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


logger = logging.get_logger(__name__)


@dataclass
class GroundedAudioDecoderOutput(ModelOutput):
    """
    Base class for outputs of the GroundedAudioDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GroundedAudioEncoderOutput(ModelOutput):
    last_hidden_state_audio: torch.FloatTensor = None
    last_hidden_state_text: torch.FloatTensor = None


@dataclass
class GroundedAudioModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    init_reference_points: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    encoder_last_hidden_state_audio: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state_text: Optional[torch.FloatTensor] = None
    enc_outputs_class: Optional[torch.FloatTensor] = None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None


@dataclass
class GroundedAudioObjectDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    init_reference_points: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state_audio: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state_text: Optional[torch.FloatTensor] = None


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->GroundedAudio
class GroundedAudioFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it user-friendly
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->GroundedAudio
def replace_batch_norm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = GroundedAudioFrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class GroundedAudioConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by GroundedAudioFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.use_timm_backbone:
            requires_backends(self, ["timm"])
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                **config.backbone_kwargs,
            )
        else:
            backbone = load_backbone(config)

        # replace batch norm by frozen batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        backbone_model_type = None
        if config.backbone is not None:
            backbone_model_type = config.backbone
        elif config.backbone_config is not None:
            backbone_model_type = config.backbone_config.model_type
        else:
            raise ValueError("Either `backbone` or `backbone_config` should be provided in the config")

        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)

    # Copied from transformers.models.detr.modeling_detr.DetrConvEncoder.forward with Detr->GroundedAudio
    def forward(self, audio_values: torch.Tensor, audio_mask: torch.Tensor):
        # send audio_values through the model to get list of feature maps
        features = self.model(audio_values) if self.config.use_timm_backbone else self.model(audio_values).feature_maps

        out = []
        for feature_map in features:
            # downsample audio_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(audio_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


# Copied from transformers.models.detr.modeling_detr.DetrConvModel with Detr->GroundedAudio
class GroundedAudioConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding, backbone_layer):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding
        self.backbone_layer = backbone_layer

    def forward(self, audio_values, audio_mask):
        _, _, out, audio_mask = self.conv_encoder(audio_values, audio_mask)
        pos = self.position_embedding(out[self.backbone_layer], audio_mask)

        return out[self.backbone_layer], pos, audio_mask


class GroundedAudioSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.d_model
        self.temperature = config.positional_embedding_temperature
        self.scale = 2 * math.pi

    def forward(self, audio_values, audio_mask):
        x_embed = audio_mask.cumsum(1, dtype=torch.float32)
        eps = 1e-6
        x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=audio_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos = x_embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos


class GroundedAudioLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, config):
        super().__init__()

        embedding_dim = config.d_model // 2
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    def forward(self, audio_values, audio_mask=None):
        height, width = audio_values.shape[-2:]
        width_values = torch.arange(width, device=audio_values.device)
        height_values = torch.arange(height, device=audio_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(audio_values.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(config):
    if config.position_embedding_type == "sine":
        position_embedding = GroundedAudioSinePositionEmbedding(config)
    elif config.position_embedding_type == "learned":
        position_embedding = GroundedAudioLearnedPositionEmbedding(config)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, seq, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_points, _ = sampling_locations.shape
    sampling_grids = (2 * sampling_locations - 1).transpose(1, 2).flatten(0, 1)
    sampling_grids = torch.cat([sampling_grids, torch.zeros_like(sampling_grids, device=sampling_grids.device)], dim=-1)

    value = value.flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, seq).unsqueeze(-1)
    sampling_value = nn.functional.grid_sample(value, sampling_grids, mode="bilinear", padding_mode="zeros", align_corners=False)
    attention_weights = attention_weights.transpose(1, 2).reshape(batch_size * num_heads, 1, num_queries, num_points)
    output = (sampling_value * attention_weights).sum(-1).view(batch_size, num_heads * hidden_dim, num_queries).transpose(1, 2).contiguous()
    return output


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention with DeformableDetr->GroundedAudio, Deformable DETR->Grounded Audio
class GroundedAudioMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: GroundedAudioConfig, num_heads: int, n_points: int):
        super().__init__()

        kernel_loaded = MultiScaleDeformableAttention is not None

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in GroundedAudioMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * n_points * 1)
        self.attention_weights = nn.Linear(config.d_model, num_heads * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.disable_custom_kernels = config.disable_custom_kernels

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_points, 1
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_points
        )
        sampling_locations = (reference_points[:, :, None, :, :1] + sampling_offsets / attention_mask.sum(-1)[:, None, None, None, None])

        output = multi_scale_deformable_attention(value, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output, attention_weights


class GroundedAudioTextEnhancerLayer(nn.Module):
    """Vanilla Transformer with text embeddings as input"""

    def __init__(self, config):
        super().__init__()
        self.self_attn = GroundedAudioMultiheadAttention(
            config, num_attention_heads=config.encoder_attention_heads // 2
        )

        # Implementation of Feedforward model
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim // 2)
        self.fc2 = nn.Linear(config.encoder_ffn_dim // 2, config.d_model)

        self.layer_norm_before = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layer_norm_after = nn.LayerNorm(config.d_model, config.layer_norm_eps)

        self.activation = ACT2FN[config.activation_function]
        self.num_heads = config.encoder_attention_heads // 2
        self.dropout = config.text_enhancer_dropout

    def with_pos_embed(self, hidden_state: Tensor, position_embeddings: Optional[Tensor]):
        return hidden_state if position_embeddings is None else hidden_state + position_embeddings

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_masks: Optional[torch.BoolTensor] = None,
        position_embeddings: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Text self-attention to enhance projection of text features generated by
        the text encoder (AutoModel based on text_config) within GroundedAudioEncoderLayer

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                Text features generated by the text encoder.
            attention_masks (`torch.BoolTensor`, *optional*):
                Attention mask for text self-attention. False for real tokens and True for padding tokens.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings to be added to the hidden states.

        Returns:
            `tuple(torch.FloatTensor)` comprising two elements:
            - **hidden_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) --
                Output of the text self-attention layer.
            - **attention_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length,
              sequence_length)`) --
                Attention weights of the text self-attention layer.
        """

        # repeat attn mask
        if attention_masks.dim() == 3 and attention_masks.shape[0] == hidden_states.shape[0]:
            # batch_size, num_queries, num_keys
            attention_masks = attention_masks[:, None, :, :]
            attention_masks = attention_masks.repeat(1, self.num_heads, 1, 1)

            dtype = hidden_states.dtype
            attention_masks = attention_masks.to(dtype=dtype)  # fp16 compatibility
            attention_masks = (1.0 - attention_masks) * torch.finfo(dtype).min

        queries = keys = self.with_pos_embed(hidden_states, position_embeddings)
        attention_output, attention_weights = self.self_attn(
            queries=queries,
            keys=keys,
            values=hidden_states,
            attention_mask=attention_masks,
            output_attentions=True,
        )
        attention_output = nn.functional.dropout(attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + attention_output
        hidden_states = self.layer_norm_before(hidden_states)

        residual = hidden_states
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual
        hidden_states = self.layer_norm_after(hidden_states)

        return hidden_states, attention_weights


class GroundedAudioBiMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        audio_dim = text_dim = config.d_model
        embed_dim = config.encoder_ffn_dim // 2
        num_heads = config.encoder_attention_heads // 2
        dropout = config.fusion_dropout

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.audio_dim = audio_dim
        self.text_dim = text_dim

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by `num_heads` (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.audio_proj = nn.Linear(self.audio_dim, self.embed_dim)
        self.text_proj = nn.Linear(self.text_dim, self.embed_dim)
        self.values_audio_proj = nn.Linear(self.audio_dim, self.embed_dim)
        self.values_text_proj = nn.Linear(self.text_dim, self.embed_dim)

        self.out_audio_proj = nn.Linear(self.embed_dim, self.audio_dim)
        self.out_text_proj = nn.Linear(self.embed_dim, self.text_dim)

    def _reshape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        audio_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        audio_attention_mask: Optional[torch.BoolTensor] = None,
        text_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Image-to-text and text-to-image cross-attention

        Args:
            audio_features (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_dim)`):
                Projected flattened image features generated by the vision backbone.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`):
                Projected text features generated by the text encoder.
            audio_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for image-to-text cross-attention. False for real tokens and True for padding tokens.
            text_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for text-to-image cross-attention. False for real tokens and True for padding tokens.

        Returns:
            `tuple(tuple(torch.FloatTensor), tuple(torch.FloatTensor))` where each inner tuple comprises an attention
            output and weights:
            - **audio_attn_output** (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_din)`)
              --
                Output of the image-to-text cross-attention layer.
            - **audio_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, vision_sequence_length,
              vision_sequence_length)`) --
                Attention weights of the image-to-text cross-attention layer.
            - **text_attn_output** (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`) --
                Output of the text-to-image cross-attention layer.
            - **text_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, text_sequence_length,
              text_sequence_length)`) --
                Attention weights of the text-to-image cross-attention layer.
        """
        batch_size, tgt_len, _ = audio_features.size()

        audio_query_states = self.audio_proj(audio_features) * self.scale
        audio_query_states = self._reshape(audio_query_states, tgt_len, batch_size)

        text_key_states = self.text_proj(text_features)
        text_key_states = self._reshape(text_key_states, -1, batch_size)

        audio_value_states = self.values_audio_proj(audio_features)
        audio_value_states = self._reshape(audio_value_states, -1, batch_size)

        text_value_states = self.values_text_proj(text_features)
        text_value_states = self._reshape(text_value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)

        audio_query_states = audio_query_states.view(*proj_shape)
        text_key_states = text_key_states.view(*proj_shape)
        audio_value_states = audio_value_states.view(*proj_shape)
        text_value_states = text_value_states.view(*proj_shape)

        src_len = text_key_states.size(1)
        attn_weights = torch.bmm(audio_query_states, text_key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (batch_size * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        attn_weights = attn_weights - attn_weights.max()
        # Do not increase -50000/50000, data type half has quite limited range
        attn_weights = torch.clamp(attn_weights, min=-50000, max=50000)

        attn_weights_transposed = attn_weights.transpose(1, 2)
        text_attn_weights = attn_weights_transposed - torch.max(attn_weights_transposed, dim=-1, keepdim=True)[0]

        # Do not increase -50000/50000, data type half has quite limited range
        text_attn_weights = torch.clamp(text_attn_weights, min=-50000, max=50000)

        # mask audio for language
        if audio_attention_mask is not None:
            audio_attention_mask = (
                audio_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            text_attn_weights.masked_fill_(audio_attention_mask, float("-inf"))

        text_attn_weights = text_attn_weights.softmax(dim=-1)

        # mask language for audio
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights.masked_fill_(text_attention_mask, float("-inf"))
        audio_attn_weights = attn_weights.softmax(dim=-1)

        audio_attn_probs = F.dropout(audio_attn_weights, p=self.dropout, training=self.training)
        text_attn_probs = F.dropout(text_attn_weights, p=self.dropout, training=self.training)

        audio_attn_output = torch.bmm(audio_attn_probs, text_value_states)
        text_attn_output = torch.bmm(text_attn_probs, audio_value_states)

        if audio_attn_output.size() != (batch_size * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`audio_attn_output` should be of size {(batch_size, self.num_heads, tgt_len, self.head_dim)}, but is {audio_attn_output.size()}"
            )

        if text_attn_output.size() != (batch_size * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`text_attn_output` should be of size {(batch_size, self.num_heads, src_len, self.head_dim)}, but is {text_attn_output.size()}"
            )

        audio_attn_output = audio_attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        audio_attn_output = audio_attn_output.transpose(1, 2)
        audio_attn_output = audio_attn_output.reshape(batch_size, tgt_len, self.embed_dim)

        text_attn_output = text_attn_output.view(batch_size, self.num_heads, src_len, self.head_dim)
        text_attn_output = text_attn_output.transpose(1, 2)
        text_attn_output = text_attn_output.reshape(batch_size, src_len, self.embed_dim)

        audio_attn_output = self.out_audio_proj(audio_attn_output)
        text_attn_output = self.out_text_proj(text_attn_output)

        return (audio_attn_output, audio_attn_weights), (text_attn_output, text_attn_weights)


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->GroundedAudio
class GroundedAudioDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class GroundedAudioFusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        drop_path = config.fusion_droppath

        # pre layer norm
        self.layer_norm_audio = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layer_norm_text = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.attn = GroundedAudioBiMultiHeadAttention(config)

        # add layer scale for training stability
        self.drop_path = GroundedAudioDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        init_values = 1e-4
        self.audio_param = nn.Parameter(init_values * torch.ones((config.d_model)), requires_grad=True)
        self.text_param = nn.Parameter(init_values * torch.ones((config.d_model)), requires_grad=True)

    def forward(
        self,
        audio_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        attention_mask_audio: Optional[torch.BoolTensor] = None,
        attention_mask_text: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        audio_features = self.layer_norm_audio(audio_features)
        text_features = self.layer_norm_text(text_features)
        (delta_a, audio_attn), (delta_t, text_attn) = self.attn(
            audio_features,
            text_features,
            audio_attention_mask=attention_mask_audio,
            text_attention_mask=attention_mask_text,
        )
        audio_features = audio_features + self.drop_path(self.audio_param * delta_a)
        text_features = text_features + self.drop_path(self.text_param * delta_t)

        return (audio_features, audio_attn), (text_features, text_attn)


class GroundedAudioDeformableLayer(nn.Module):
    def __init__(self, config: GroundedAudioConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = GroundedAudioMultiscaleDeformableAttention(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, attn_weights


# Based on https://github.com/IDEA-Research/GroundedAudio/blob/2b62f419c292ca9c518daae55512fabc3fead4a4/GroundedAudio/models/GroundedAudio/utils.py#L24
def get_sine_pos_embed(
    pos_tensor: torch.Tensor, num_pos_feats: int = 128, temperature: int = 10000, exchange_xy: bool = True
) -> Tensor:
    """
    Generate sine position embeddings from a position tensor.

    Args:
        pos_tensor (torch.Tensor):
            Tensor containing positions. Shape: [..., n].
        num_pos_feats (`int`, *optional*, defaults to 128):
            Projected shape for each float in the tensor.
        temperature (`int`, *optional*, defaults to 10000):
            Temperature in the sine/cosine function.
        exchange_xy (`bool`, *optional*, defaults to `True`):
            Exchange pos x and pos y. For example, input tensor is [x,y], the results will be [pos(y), pos(x)].

    Returns:
        position_embeddings (torch.Tensor): shape: [..., n * hidden_size].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_tensor = pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)
    position_embeddings = [sine_func(x) for x in pos_tensor]
    if exchange_xy:
        position_embeddings[0], position_embeddings[1] = position_embeddings[1], position_embeddings[0]
    position_embeddings = torch.cat(position_embeddings, dim=-1)
    return position_embeddings


class GroundedAudioEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.d_model = config.d_model

        self.text_enhancer_layer = GroundedAudioTextEnhancerLayer(config)
        self.fusion_layer = GroundedAudioFusionLayer(config)
        # self.deformable_layer = GroundedAudioDeformableLayer(config)
        self.audio_enhance_layer = GroundedAudioTextEnhancerLayer(config)

    def get_text_position_embeddings(
        self,
        text_features: Tensor,
        text_position_embedding: Optional[torch.Tensor],
        text_position_ids: Optional[torch.Tensor],
    ) -> Tensor:
        batch_size, seq_length, _ = text_features.shape
        if text_position_embedding is None and text_position_ids is None:
            text_position_embedding = torch.arange(seq_length, device=text_features.device)
            text_position_embedding = text_position_embedding.float()
            text_position_embedding = text_position_embedding.unsqueeze(0).unsqueeze(-1)
            text_position_embedding = text_position_embedding.repeat(batch_size, 1, 1)
            text_position_embedding = get_sine_pos_embed(
                text_position_embedding, num_pos_feats=self.d_model, exchange_xy=False
            )
        if text_position_ids is not None:
            text_position_embedding = get_sine_pos_embed(
                text_position_ids[..., None], num_pos_feats=self.d_model, exchange_xy=False
            )

        return text_position_embedding

    def forward(
        self,
        audio_features: Tensor,
        audio_position_embedding: Tensor,
        key_padding_mask: Tensor,
        audio_self_attention_maks: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
    ):
        text_position_embedding = self.get_text_position_embeddings(
            text_features, text_position_embedding, text_position_ids
        )

        (audio_features, vision_fused_attn), (text_features, text_fused_attn) = self.fusion_layer(
            audio_features=audio_features,
            text_features=text_features,
            attention_mask_audio=key_padding_mask,
            attention_mask_text=text_attention_mask,
        )

        (text_features, text_enhanced_attn) = self.text_enhancer_layer(
            hidden_states=text_features,
            attention_masks=~text_self_attention_masks,  # note we use ~ for mask here
            position_embeddings=(text_position_embedding if text_position_embedding is not None else None),
        )

        (audio_features, audio_deformable_attn) = self.audio_enhance_layer(
            hidden_states=audio_features,
            attention_masks=~audio_self_attention_maks,
            position_embeddings=audio_position_embedding
        )

        return (
            (audio_features, text_features),
            (vision_fused_attn, text_fused_attn, text_enhanced_attn, audio_deformable_attn),
        )


class GroundedAudioMultiheadAttention(nn.Module):
    """Equivalent implementation of nn.MultiheadAttention with `batch_first=True`."""

    def __init__(self, config, num_attention_heads=None):
        super().__init__()
        if config.hidden_size % num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(config.hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(queries))
        key_layer = self.transpose_for_scores(self.key(keys))
        value_layer = self.transpose_for_scores(self.value(values))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in GroundedAudioModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.out_proj(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class GroundedAudioDecoderLayer(nn.Module):
    def __init__(self, config: GroundedAudioConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self-attention
        self.self_attn = GroundedAudioMultiheadAttention(config, num_attention_heads=config.decoder_attention_heads)

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # cross-attention text
        self.encoder_attn_text = GroundedAudioMultiheadAttention(
            config, num_attention_heads=config.decoder_attention_heads
        )
        self.encoder_attn_text_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # cross-attention
        self.encoder_attn = GroundedAudioMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        audio_encoder_hidden_states: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        text_encoder_hidden_states: Optional[torch.Tensor] = None,
        text_encoder_attention_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states

        # Self Attention
        queries = keys = self.with_pos_embed(hidden_states, position_embeddings)
        hidden_states, self_attn_weights = self.self_attn(
            queries=queries,
            keys=keys,
            values=hidden_states,
            attention_mask=self_attn_mask,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention Text
        queries = self.with_pos_embed(hidden_states, position_embeddings)
        hidden_states, text_cross_attn_weights = self.encoder_attn_text(
            queries=queries,
            keys=text_encoder_hidden_states,
            values=text_encoder_hidden_states,
            attention_mask=text_encoder_attention_mask,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states
        hidden_states = self.encoder_attn_text_layer_norm(hidden_states)

        third_residual = hidden_states

        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=audio_encoder_attention_mask,
            encoder_hidden_states=audio_encoder_hidden_states,
            encoder_attention_mask=audio_encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = third_residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, text_cross_attn_weights, cross_attn_weights)

        return outputs


class GroundedAudioContrastiveEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_text_len = config.max_text_len

    def forward(
        self,
        audio_hidden_state: torch.FloatTensor,
        text_hidden_state: torch.FloatTensor,
        text_token_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        output = audio_hidden_state @ text_hidden_state.transpose(-1, -2)
        output = output.masked_fill(~text_token_mask[:, None, :], float("-inf"))

        # padding to max_text_len
        new_output = torch.full((*output.shape[:-1], self.max_text_len), float("-inf"), device=output.device)
        new_output[..., : output.shape[-1]] = output

        return new_output


class GroundedAudioPreTrainedModel(PreTrainedModel):
    config_class = GroundedAudioConfig
    base_model_prefix = "model"
    main_input_name = "audio_values"

    def _init_weights(self, module):
        std = self.config.init_std

        if isinstance(module, GroundedAudioLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, GroundedAudioMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads // 2, dtype=torch.int64).to(default_dtype) * (
                2 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 1)
                .repeat(1, 1, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
        elif isinstance(module, GroundedAudioBiMultiHeadAttention):
            nn.init.xavier_uniform_(module.audio_proj.weight)
            module.audio_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.text_proj.weight)
            module.text_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.values_audio_proj.weight)
            module.values_audio_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.values_text_proj.weight)
            module.values_text_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.out_audio_proj.weight)
            module.out_audio_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.out_text_proj.weight)
            module.out_text_proj.bias.data.fill_(0)
        elif isinstance(module, (GroundedAudioEncoderLayer, GroundedAudioDecoderLayer)):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=std)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GroundedAudioMLPPredictionHead):
            nn.init.constant_(module.layers[-1].weight.data, 0)
            nn.init.constant_(module.layers[-1].bias.data, 0)

        if hasattr(module, "reference_points") and not self.config.two_stage:
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GroundedAudioDecoder):
            module.gradient_checkpointing = value


class GroundedAudioEncoder(GroundedAudioPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`GroundedAudioEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: GroundedAudioConfig
    """

    def __init__(self, config: GroundedAudioConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([GroundedAudioEncoderLayer(config) for _ in range(config.encoder_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            # TODO: valid_ratios could be useless here. check https://github.com/fundamentalvision/Deformable-DETR/issues/36
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        audio_features: Tensor,
        audio_attention_mask: Tensor,
        audio_position_embedding: Tensor,
        audio_self_attention_maks: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
        return_dict=None,
    ):

        for i, encoder_layer in enumerate(self.layers):
            (audio_features, text_features), attentions = encoder_layer(
                audio_features=audio_features,
                audio_position_embedding=audio_position_embedding,
                audio_self_attention_maks=audio_self_attention_maks,
                key_padding_mask=audio_attention_mask,
                text_features=text_features,
                text_attention_mask=text_attention_mask,
                text_position_embedding=text_position_embedding,
                text_self_attention_masks=text_self_attention_masks,
                text_position_ids=text_position_ids,
            )

        if not return_dict:
            enc_outputs = [audio_features, text_features]
            return tuple(v for v in enc_outputs if v is not None)
        return GroundedAudioEncoderOutput(
            last_hidden_state_audio=audio_features,
            last_hidden_state_text=text_features,
        )


class GroundedAudioDecoder(GroundedAudioPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`GroundedAudioDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some tweaks for Grounded Audio:

    - `position_embeddings`, `reference_points`, `spatial_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.

    Args:
        config: GroundedAudioConfig
    """

    def __init__(self, config: GroundedAudioConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layers = nn.ModuleList([GroundedAudioDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.reference_points_head = GroundedAudioMLPPredictionHead(
            config.query_dim // 2 * config.d_model, config.d_model, config.d_model, 2
        )
        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement as in two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.query_scale = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds,
        audio_encoder_hidden_states,
        audio_encoder_attention_mask=None,
        text_encoder_hidden_states=None,
        text_encoder_attention_mask=None,
        reference_points=None,
        self_attn_mask=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        intermediate = ()
        intermediate_reference_points = ()

        if text_encoder_attention_mask is not None:
            dtype = text_encoder_hidden_states.dtype

            text_encoder_attention_mask = text_encoder_attention_mask[:, None, None, :]
            text_encoder_attention_mask = text_encoder_attention_mask.repeat(
                1, self.config.decoder_attention_heads, self.config.num_queries, 1
            )
            text_encoder_attention_mask = text_encoder_attention_mask.to(dtype=dtype)
            text_encoder_attention_mask = text_encoder_attention_mask * torch.finfo(dtype).min

        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            query_pos = get_sine_pos_embed(reference_points_input[:, :, 0, :], num_pos_feats=self.config.d_model // 2)
            query_pos = self.reference_points_head(query_pos)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, True)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    query_pos,
                    reference_points_input,
                    audio_encoder_hidden_states,
                    audio_encoder_attention_mask,
                    text_encoder_hidden_states,
                    text_encoder_attention_mask,
                    self_attn_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=query_pos,
                    reference_points=reference_points_input,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                    text_encoder_hidden_states=text_encoder_hidden_states,
                    text_encoder_attention_mask=text_encoder_attention_mask,
                    self_attn_mask=self_attn_mask,
                )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                new_reference_points = tmp
                new_reference_points = tmp + torch.special.logit(reference_points, eps=1e-5)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            intermediate += (self.layer_norm(hidden_states),)
            intermediate_reference_points += (reference_points,)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        hidden_states = self.layer_norm(hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                ]
                if v is not None
            )
        return GroundedAudioDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
        )


# these correspond to [CLS], [SEP], . and ?
SPECIAL_TOKENS = [101, 102, 1012, 1029]


def generate_masks_with_special_tokens_and_transfer_map(input_ids: torch.LongTensor) -> Tuple[Tensor, Tensor]:
    """Generate attention mask between each pair of special tokens and positional ids.
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
    Returns:
        `tuple(torch.Tensor)` comprising attention mask between each special tokens and position_ids:
        - **attention_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length, sequence_length)`)
        - **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
    """
    batch_size, num_token = input_ids.shape
    # special_tokens_mask: batch_size, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((batch_size, num_token), device=input_ids.device).bool()
    for special_token in SPECIAL_TOKENS:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(batch_size, 1, 1)
    position_ids = torch.zeros((batch_size, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if col == 0:
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        elif col == num_token - 1:
            attention_mask[row, previous_col + 1 : col, previous_col + 1 : col] = True
            position_ids[row, previous_col + 1 : col] = torch.arange(0, col - previous_col - 1, device=input_ids.device)
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(0, col - previous_col, device=input_ids.device)

        previous_col = col

    return attention_mask, position_ids.to(torch.long)


class GroundedAudioModel(GroundedAudioPreTrainedModel):
    def __init__(self, config: GroundedAudioConfig):
        super().__init__(config)

        # Create backbone + positional encoding
        backbone = SenseVoiceSmall(config.backbone_config)
        if config.backbone_load is not None and os.path.exists(config.backbone_load):
            backbone.load_state_dict(torch.load(config.backbone_load))
        position_embeddings = build_position_encoding(config)
        self.backbone = GroundedAudioConvModel(backbone, position_embeddings, config.backbone_layer)

        self.input_proj_vision = nn.Sequential(
            nn.Linear(backbone.encoder_output_size, config.d_model),
            nn.LayerNorm(config.d_model, config.layer_norm_eps),
        )

        self.text_backbone = AutoModel.from_config(
            config.text_config, add_pooling_layer=False, attn_implementation=config._attn_implementation
        )
        if config.text_backbone_load is not None and os.path.exists(config.text_backbone_load):
            self.text_backbone.load_state_dict(torch.load(config.text_backbone_load))
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.d_model)

        if config.embedding_init_target or not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        self.encoder = GroundedAudioEncoder(config)
        self.decoder = GroundedAudioDecoder(config)

        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
            if (
                config.two_stage_bbox_embed_share
                and config.decoder_bbox_embed_share
                and self.decoder.bbox_embed is not None
            ):
                self.encoder_output_bbox_embed = self.decoder.bbox_embed
            else:
                self.encoder_output_bbox_embed = GroundedAudioMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=2, num_layers=3
                )

            self.encoder_output_class_embed = GroundedAudioContrastiveEmbedding(config)
        else:
            self.reference_points = nn.Embedding(config.num_queries, 2)

        self.config.backbone_load = None
        self.config.text_backbone_load = None
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for param in self.backbone.conv_encoder.parameters():
            param.requires_grad_(False)
        for param in self.text_backbone.parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for param in self.backbone.conv_encoder.parameters():
            param.requires_grad_(True)
        for param in self.text_backbone.parameters():
            param.requires_grad_(True)

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def generate_encoder_output_proposals(self, enc_output, padding_mask):
        batch_size, seq_len = enc_output.shape[:2]
        grid = torch.linspace(0, seq_len-1, seq_len, dtype=torch.float32, device=enc_output.device).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        scale = torch.sum(~padding_mask, dim=1).unsqueeze(-1).unsqueeze(-1)
        grid = grid / scale
        width = torch.ones_like(grid) * 0.05
        # gird: begin point, width: end point, 0.05: 距离初始值
        proposal = torch.cat((grid, width), -1).view(batch_size, -1, 2)
        # proposal: [bs, time_len, 2], 每个时间点初始预测的起始事件时间和结束事件时间

        proposal_val = ((proposal > 0.01) & (proposal < 0.99)).all(-1, keepdim=True)
        proposal = torch.log(proposal / (1-proposal))  # inverse sigmoid: 不理解
        proposal = proposal.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        proposal = proposal.masked_fill(~proposal_val, float("inf")) # 个人理解不用做这个, 在图像中, 这里是为了避免在图像边界的boundingbox预测.

        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~proposal_val, float(0))
        object_query = self.enc_output_norm(self.enc_output(object_query))

        return object_query, proposal

    def forward(
        self,
        audio_values: Tensor,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        audio_mask: Optional[Tensor] = None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(input_ids)
        # audio_self_attention_masks, position_ids

        text_token_mask = attention_mask.bool()  # just to avoid renaming everywhere

        max_text_len = self.config.max_text_len
        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
            position_ids = position_ids[:, :max_text_len]
            input_ids = input_ids[:, :max_text_len]
            token_type_ids = token_type_ids[:, :max_text_len]
            text_token_mask = text_token_mask[:, :max_text_len]

        # Extract text features from text backbone
        text_outputs = self.text_backbone(input_ids, text_self_attention_masks, token_type_ids, position_ids, return_dict=return_dict)
        text_features = text_outputs.last_hidden_state if return_dict else text_outputs[0]
        text_features = self.text_projection(text_features)

        batch_size, audio_len, embedding_dim = audio_values.shape

        audio_features, position_embeddings, audio_mask = self.backbone(audio_values, audio_mask)
        audio_features = self.input_proj_vision(audio_features)

        query_embeds = None
        if self.config.embedding_init_target or self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight

        if encoder_outputs is None:
            audio_self_mask = audio_mask.unsqueeze(1)
            encoder_outputs = self.encoder(
                audio_features=audio_features,
                audio_attention_mask=~audio_mask.bool(),
                audio_self_attention_maks=~audio_self_mask.bool(),
                audio_position_embedding=position_embeddings,
                text_features=text_features,
                text_attention_mask=~text_token_mask,
                text_position_embedding=None,
                text_self_attention_masks=~text_self_attention_masks,
                text_position_ids=position_ids,
                return_dict=return_dict,
            )
        
        # additional code end

        # If the user passed a tuple for encoder_outputs, we wrap it in a GroundedAudioEncoderOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, GroundedAudioEncoderOutput):
            encoder_outputs = GroundedAudioEncoderOutput(
                last_hidden_state_audio=encoder_outputs[0],
                last_hidden_state_text=encoder_outputs[1],
                audio_hidden_states=encoder_outputs[2] if output_hidden_states else None,
                text_hidden_states=encoder_outputs[3] if output_hidden_states else None,
                attentions=encoder_outputs[-1] if output_attentions else None,
            )

        # Fifth, prepare decoder inputs
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        if self.config.two_stage:
            object_query_embedding, output_proposals = self.generate_encoder_output_proposals(encoder_outputs[0], ~audio_mask.bool())

            # hack implementation as in two-stage Deformable DETR
            # apply a detection head to each pixel (A.4 in paper)
            # linear projection for bounding box binary classification (i.e. foreground and background)
            enc_outputs_class = self.encoder_output_class_embed(object_query_embedding, encoder_outputs[1], text_token_mask)            
            # 3-layer FFN to predict bounding boxes coordinates (bbox regression branch)
            delta_bbox = self.encoder_output_bbox_embed(object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals

            # only keep top scoring `config.num_queries` proposals
            topk = self.config.num_queries
            topk_logits = enc_outputs_class.max(-1)[0]
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
            topk_coords_logits = torch.gather(
                enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 2)
            )

            topk_coords_logits = topk_coords_logits.detach()
            reference_points = topk_coords_logits.sigmoid()
            init_reference_points = reference_points
            if query_embeds is not None:
                target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                target = torch.gather(
                    object_query_embedding, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
                ).detach()
        else:
            target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            reference_points = self.reference_points.weight.unsqueeze(0).repeat(batch_size, 1, 1).sigmoid()
            init_reference_points = reference_points

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            audio_encoder_hidden_states=encoder_outputs[0],
            audio_encoder_attention_mask=audio_mask.bool(),
            text_encoder_hidden_states=encoder_outputs[1],
            text_encoder_attention_mask=~text_token_mask,
            reference_points=reference_points,
            self_attn_mask=None,
            return_dict=return_dict,
        )

        if not return_dict:
            enc_outputs = tuple(value for value in [enc_outputs_class, enc_outputs_coord_logits] if value is not None)
            tuple_outputs = (
                (decoder_outputs[0], init_reference_points) + decoder_outputs[1:] + encoder_outputs + enc_outputs
            )

            return tuple_outputs

        return GroundedAudioModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            init_reference_points=init_reference_points,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            encoder_last_hidden_state_audio=encoder_outputs.last_hidden_state_audio,
            encoder_last_hidden_state_text=encoder_outputs.last_hidden_state_text,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits
        )


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class GroundedAudioMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return boxes[:, 1] - boxes[:, 0]


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    right = torch.min(boxes1[:, None, 1], boxes2[:, 1])

    inter = (right - left).clamp(min=0)
    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    iou, union = box_iou(boxes1, boxes2)

    top = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    bottom = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    area = (bottom - top).clamp(min=0)
    return iou - (area - union) / area


# Copied from transformers.models.detr.modeling_detr._max_by_axis
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# Copied from transformers.models.detr.modeling_detr.dice_loss
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


# Copied from transformers.models.detr.modeling_detr.sigmoid_focal_loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy(prob, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# Copied from transformers.models.detr.modeling_detr.NestedTensor
class NestedTensor:
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# Copied from transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)


def center_to_corners_one_dim(bbox):
    center, width = bbox.unbind(-1)
    bbox_corners = torch.stack([(center - width), (center + width)], dim=-1)
    return bbox_corners


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrHungarianMatcher with DeformableDetr->GroundedAudio
class GroundedAudioHungarianMatcher(nn.Module):
    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes].
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and boxes [batch size, num_labels, len_labels] / [batch size, num_labels]
        target_ids = torch.cat([v["class_labels"] for v in targets])  # [batch size*num_labels, len_labels] / [batch size*num_labels].
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        # num labels: [num_labels, label_len]: label_len不同尺寸1, 2, 3, 4, 5, 6, 7, 8
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())  # [batch_size, num_queries, num_classes]
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())  # [batch_size, num_queries, num_classes]
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]  # [batch size*num_query, batch size*num_labels, len_labels]
        # # [batch size*num_query, batch size*num_labels*len_labels]

        # Compute the L1 cost between boxes: 不用管batch, 下面分开就好.
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -box_iou(center_to_corners_one_dim(out_bbox), center_to_corners_one_dim(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss with DeformableDetr->GroundedAudio
class GroundedAudioLoss(nn.Module):
    def __init__(self, matcher, num_classes, focal_alpha, losses):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.losses = losses

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        # joblib.dump(source_logits, "./temp/source_logits.joblib")
        # joblib.dump(targets, "./temp/targets.joblib")
        # joblib.dump(indices, "./temp/indices.joblib")
        # exit()

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    # Copied from transformers.models.detr.modeling_detr.DetrLoss.loss_cardinality
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    # Copied from transformers.models.detr.modeling_detr.DetrLoss.loss_boxes
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_iou(center_to_corners_one_dim(source_boxes), center_to_corners_one_dim(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    # Copied from transformers.models.detr.modeling_detr.DetrLoss._get_source_permutation_idx
    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    # Copied from transformers.models.detr.modeling_detr.DetrLoss._get_target_permutation_idx
    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs" and k != "enc_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                world_size = PartialState().num_processes
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class GroundedAudioForObjectDetection(GroundedAudioPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    # the bbox_embed in the decoder are all clones though
    _tied_weights_keys = [r"bbox_embed\.[1-9]\d*", r"model\.decoder\.bbox_embed\.[0-9]\d*"]

    def __init__(self, config: GroundedAudioConfig):
        super().__init__(config)

        self.model = GroundedAudioModel(config)
        _class_embed = GroundedAudioContrastiveEmbedding(config)

        if config.decoder_bbox_embed_share:
            _bbox_embed = GroundedAudioMLPPredictionHead(
                input_dim=config.d_model, hidden_dim=config.d_model, output_dim=2, num_layers=3
            )
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        else:
            for _ in range(config.decoder_layers):
                _bbox_embed = GroundedAudioMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=2, num_layers=3
                )
                self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])
        # hack for box-refinement
        self.model.decoder.bbox_embed = self.bbox_embed
        # hack implementation for two-stage
        self.model.decoder.class_embed = self.class_embed

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(
        self,
        audio_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        audio_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Union[GroundedAudioEncoderOutput, Tuple]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: List[Dict[str, Union[torch.LongTensor, torch.FloatTensor]]] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # First, sent images through Grounded Audio base model to obtain encoder + decoder outputs
        outputs = self.model(
            audio_values=audio_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            audio_mask=audio_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        enc_text_hidden_state = outputs.encoder_last_hidden_state_text if return_dict else outputs[5]
        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference_points = outputs.init_reference_points if return_dict else outputs[1]
        inter_references_points = outputs.intermediate_reference_points if return_dict else outputs[3]

        # class logits + predicted bounding boxes
        outputs_classes = []
        outputs_coords = []

        # hidden_states are of shape (batch_size, num_stages, height, width)
        # predict class and bounding box deltas for each stage
        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            if level == 0:
                reference = init_reference_points
            else:
                reference = inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)
            outputs_class = self.class_embed[level](
                audio_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=attention_mask.bool(),
            )
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])         
            outputs_coord_logits = delta_bbox + reference

            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = GroundedAudioHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = GroundedAudioLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                focal_alpha=self.config.focal_alpha,
                losses=losses,
            )
            criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            if self.config.auxiliary_loss:
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs
            if self.config.two_stage:
                enc_outputs_coord = outputs[-1].sigmoid()
                outputs_loss["enc_outputs"] = {"logits": outputs[-2], "pred_boxes": enc_outputs_coord}

            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output

            return tuple_outputs

        dict_outputs = GroundedAudioObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            last_hidden_state=outputs.last_hidden_state,
            auxiliary_outputs=auxiliary_outputs,
            encoder_last_hidden_state_audio=outputs.encoder_last_hidden_state_audio,
            encoder_last_hidden_state_text=outputs.encoder_last_hidden_state_text,
            
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
        )

        return dict_outputs

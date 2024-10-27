# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Grounding Audio model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.utils.backbone_utils import verify_backbone_config_arguments
from transformers import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class SpecAugConfig(PretrainedConfig):
    def __init__(
            self, 
            apply_freq_mask=True,
            apply_time_mask=True, 
            apply_time_warp=False,
            freq_mask_width_range=None,
            lfr_rate=None,
            num_freq_mask=None,
            num_time_mask=None,
            time_mask_width_range=None,
            time_warp_mode=None,
            time_warp_window=None,
            **kwargs
        ):
        self.apply_freq_mask = apply_freq_mask
        self.apply_time_mask = apply_time_mask
        self.apply_time_warp = apply_time_warp
        self.freq_mask_width_range = freq_mask_width_range
        self.lfr_rate = lfr_rate
        self.num_freq_mask = num_freq_mask
        self.num_time_mask = num_time_mask
        self.time_mask_width_range = time_mask_width_range
        self.time_warp_mode = time_warp_mode
        self.time_warp_window = time_warp_window
        super().__init__(**kwargs)


class SenseVoiceEncoderConfig(PretrainedConfig):
    def __init__(
            self, 
            attention_dropout_rate=True,
            attention_heads=True, 
            dropout_rate=False,
            kernel_size=None,
            linear_units=None,
            normalize_before=True,
            num_blocks=None,
            output_size=None,
            sanm_shfit=None,
            tp_blocks=None,
            input_size=None,
            **kwargs
        ):
        self.attention_dropout_rate = attention_dropout_rate
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.linear_units = linear_units
        self.normalize_before = normalize_before
        self.num_blocks = num_blocks
        self.output_size = output_size
        self.sanm_shfit = sanm_shfit
        self.tp_blocks = tp_blocks
        self.input_size = input_size
        super().__init__(**kwargs)


class SenseVoiceConfig(PretrainedConfig):
    def __init__(
            self, 
            specaug=None,
            specaug_conf=None,
            encoder_conf=None,
            normalize=None,
            length_normalized_loss=True,
            input_size=None,
            ignore_id=None,
            **kwargs
        ):
        self.specaug = specaug
        self.specaug_conf = SpecAugConfig(**specaug_conf)
        self.encoder_conf = SenseVoiceEncoderConfig(**encoder_conf)
        self.normalize = normalize
        self.length_normalized_loss = length_normalized_loss
        self.input_size = input_size
        self.ignore_id = ignore_id
        super().__init__(**kwargs)


class GroundingAudioConfig(PretrainedConfig):
    model_type = "grounding-audio"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        backbone_config=None,
        text_config=None,
        backbone_layer=50,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        num_queries=900,
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        auxiliary_loss=False,
        position_embedding_type="sine",
        num_feature_levels=4,
        encoder_n_points=4,
        decoder_n_points=4,
        two_stage=True,
        class_cost=1.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        bbox_loss_coefficient=5.0,
        giou_loss_coefficient=2.0,
        focal_alpha=0.25,
        disable_custom_kernels=False,
        # other parameters
        max_text_len=256,
        text_enhancer_dropout=0.0,
        fusion_droppath=0.1,
        fusion_dropout=0.0,
        embedding_init_target=True,
        query_dim=4,
        decoder_bbox_embed_share=True,
        two_stage_bbox_embed_share=False,
        positional_embedding_temperature=20,
        init_std=0.02,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        # Audio encoder backbone
        self.backbone_config = SenseVoiceConfig(**backbone_config)
        # Text encoder backbone
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "bert"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["bert"]()
        self.text_config = text_config

        self.backbone_layer = backbone_layer
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.num_queries = num_queries
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.auxiliary_loss = auxiliary_loss
        self.position_embedding_type = position_embedding_type
        # deformable attributes
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.two_stage = two_stage
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels
        self.max_text_len = max_text_len

        # Text Enhancer
        self.text_enhancer_dropout = text_enhancer_dropout
        # Fusion
        self.fusion_droppath = fusion_droppath
        self.fusion_dropout = fusion_dropout
        # Others
        self.embedding_init_target = embedding_init_target
        self.query_dim = query_dim
        self.decoder_bbox_embed_share = decoder_bbox_embed_share
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        self.positional_embedding_temperature = positional_embedding_temperature
        self.init_std = init_std
        self.layer_norm_eps = layer_norm_eps
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model

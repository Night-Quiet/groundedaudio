# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""
Processor class for Grounding Audio.
"""

from typing import Unpack

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.utils import logging
from transformers import FEATURE_EXTRACTOR_MAPPING
from .feature_extraction_grounding_audio import GroundingAudioFeatureExtractor
FEATURE_EXTRACTOR_MAPPING.register("GroundingAudioFeatureExtractor", GroundingAudioFeatureExtractor)


logger = logging.get_logger(__name__)


class GroundingAudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True, 
            "truncation": True, 
            "return_tensors": "pt"
        },
        "audio_kwargs": {},
    }


class GroundingAudioProcessor(ProcessorMixin):
    attributes = ["audio_processor", "tokenizer"]
    
    audio_processor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        audio_processor=None,
        tokenizer=None,
    ):
        super().__init__(audio_processor, tokenizer)

    def __call__(
        self,
        audios=None,
        text=None,
        **kwargs: Unpack[GroundingAudioProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            GroundingAudioProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        audios_input = self.audio_processor(audios, **output_kwargs["audio_kwargs"])
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **audios_input})

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + audio_processor_input_names))

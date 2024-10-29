# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Feature extractor class for Grounded Audio
"""

from typing import List, Optional, Union

import copy
import json
import numpy as np

from transformers import is_torch_available
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging
from targetedmind.modules.sensevoice.utils import load_audio_text_image_video, extract_fbank
from targetedmind.modules.sensevoice.frontend import WavFrontend


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)

class GroundedAudioFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["audio_values"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        frontend_config=None,
        data_type="sound",
        padding_value=0.0,
        return_attention_mask=True,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.sampling_rate = sampling_rate
        self.frontend = WavFrontend(**frontend_config)
        self.data_type = data_type
        
    def __call__(
        self,
        raw_speech: Optional[list],
    ) -> BatchFeature:
        speech, speech_lengths = extract_fbank(raw_speech, data_type=self.data_type, frontend=self.frontend)

        if self.return_attention_mask:
            speech_mask = torch.arange(0, speech.shape[1]).unsqueeze(0).repeat(speech.shape[0], 1)
            speech_mask = (speech_mask < speech_lengths.unsqueeze(1)).int()
            batched_speech = BatchFeature({"audio_values": speech, "audio_mask": speech_mask})
        else:
            batched_speech = BatchFeature({"audio_values": speech})
        return batched_speech


    def to_json_string(self) -> str:
        dictionary = self.to_dict()
        dictionary["frontend_config"] = copy.deepcopy({key: value for key, value in self.frontend.__dict__.items() if not key.startswith("_")})
        del dictionary["frontend"]
        del dictionary["frontend_config"]["training"]

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"


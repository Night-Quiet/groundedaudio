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
Processor class for Grounded Audio.
"""

import torch
from typing import Unpack

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.utils import logging
from transformers import FEATURE_EXTRACTOR_MAPPING
from .feature_extraction_grounded_audio import GroundedAudioFeatureExtractor
from .grounded_audio_model import center_to_corners_one_dim
FEATURE_EXTRACTOR_MAPPING.register("GroundedAudioFeatureExtractor", GroundedAudioFeatureExtractor)


logger = logging.get_logger(__name__)


def get_phrases_from_posmap(posmaps, input_ids):
    left_idx = 0
    right_idx = posmaps.shape[-1] - 1

    # Avoiding altering the input tensor
    posmaps = posmaps.clone()

    posmaps[:, 0 : left_idx + 1] = False
    posmaps[:, right_idx:] = False

    token_ids = []
    for posmap in posmaps:
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids.append([input_ids[i] for i in non_zero_idx])

    return token_ids


class GroundedAudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True, 
            "truncation": True, 
            "return_tensors": "pt"
        },
        "audio_kwargs": {},
    }


class GroundedAudioProcessor(ProcessorMixin):
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
        **kwargs: Unpack[GroundedAudioProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            GroundedAudioProcessorKwargs,
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


    def post_process_grounded_object_detection(
        self,
        outputs,
        input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_size=None,
    ):
        logits, boxes = outputs.logits, outputs.pred_boxes

        probs = torch.sigmoid(logits)  # (batch_size, num_queries, 256)
        scores = torch.max(probs, dim=-1)[0]  # (batch_size, num_queries)

        if target_size is not None:
            boxes = boxes * target_size
        boxes = center_to_corners_one_dim(boxes)

        results = []
        for idx, (s, b, p) in enumerate(zip(scores, boxes, probs)):
            score = s[s > box_threshold]
            box = b[s > box_threshold]
            prob = p[s > box_threshold]
            label_ids = get_phrases_from_posmap(prob > text_threshold, input_ids[idx])
            label = self.tokenizer.batch_decode(label_ids)
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

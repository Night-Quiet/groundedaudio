import os
import wave
import json
import torch
import joblib
import shutil
from pathlib import Path
from groundedaudio.sensevoice_model import SenseVoiceSmall
from transformers import BertModel, BertTokenizer
from groundedaudio.configuration_grounded_audio import GroundedAudioConfig

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import torchaudio


import random

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

import requests
from PIL import Image
from datasets import load_dataset

from groundedaudio.processing_grounded_audio import GroundedAudioProcessor

from groundedaudio.grounded_audio_model import GroundedAudioModel, GroundedAudioForObjectDetection
from groundedaudio.configuration_grounded_audio import GroundedAudioConfig
from groundedaudio.processing_grounded_audio import GroundedAudioProcessor
from groundedaudio.sensevoice_model import SenseVoiceSmall

from transformers import AutoProcessor, GroundingDinoForObjectDetection
from utils import AudioSetSLPreprocessor


model_id = "/root/autodl-tmp/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
text = "a cat. a remote control."
inputs = processor(images=[image, image], text=[text, text], return_tensors="pt").to(device)

targets = []
bs, seq_len = inputs.input_ids.shape

for _ in range(bs):
    target = {
        "class_labels": torch.tensor([1, 5], device=device, dtype=torch.long),
        "boxes": torch.tensor([[344.8151,  23.1787, 637.3994, 373.8305], [ 11.9177,  51.5846, 316.5746, 472.8922]], device=device)
    }
    targets.append(target)

with torch.no_grad():
    outputs = model(**inputs, labels=targets)

print(outputs.loss)

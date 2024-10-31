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


def groundeddino_inference_demo():
    # from grounded_audio_model_origin import GroundedDinoModel
    # from transformers import AutoProcessor

    model_id = "/root/autodl-tmp/grounded-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
            "class_labels": torch.tensor([3, 5], device=device, dtype=torch.long),
            "boxes": torch.tensor([[344.8151,  23.1787, 637.3994, 373.8305], [ 11.9177,  51.5846, 316.5746, 472.8922]], device=device)
        }
        targets.append(target)

    with torch.no_grad():
        outputs = model(**inputs, labels=targets)
    
    print(outputs.loss)

 
def groundedaudio_inference_demo():
    from targetedmind.modules.sensevoice.utils import load_audio_text_image_video
    
    file_path = ["/root/autodl-tmp/audioset_strong/val/YP996TZp25PM.wav", "/root/autodl-tmp/audioset_strong/val/Yt7qb5pBcBY4.wav"]
    sentences = ["this is a speech. this is a output. this is a audio. this is a output. this is a audio", "this is a output. this is a audio. this is a output. this is a audio. this is a output. this is a audio"]
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    config = GroundedAudioConfig.from_json_file("/root/groundedaudio_pretrained/config.json")
    processor = GroundedAudioProcessor.from_pretrained("/root/groundedaudio_pretrained")

    inputs = processor(audios=load_audio_text_image_video(file_path, fs=32000), text=sentences)
    
    targets = []
    bs, seq_len = inputs.input_ids.shape

    for _ in range(bs):
        target = {
            "class_labels": torch.randint(0, 10, (5,), device=device, dtype=torch.long),
            "boxes": torch.randn(5, 2, device=device)
        }
        targets.append(target)

    model = GroundedAudioForObjectDetection(config).to(device)
    model_output = model(
        audio_values=inputs.audio_values,
        audio_mask=inputs.audio_mask,
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        token_type_ids=inputs.token_type_ids,
        labels=targets
    )
    print(model_output[0])
    pass


def demo():
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    from PIL import Image
    import requests

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open("11.jpg")
    text = "blue cloth. red cloth"

    processor = AutoProcessor.from_pretrained("/root/autodl-tmp/grounded-dino-tiny")
    model = GroundingDinoForObjectDetection.from_pretrained("/root/autodl-tmp/grounded-dino-tiny")

    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.image_processor.post_process_object_detection(
        outputs, threshold=0.35, target_sizes=target_sizes
    )[0]
    # print(results["scores"], results["labels"], results["boxes"])
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 1) for i in box.tolist()]
        print(f"Detected {label.item()} with confidence " f"{round(score.item(), 2)} at location {box}")
    pass


if __name__ == "__main__":
    groundedaudio_inference_demo()
    # print(torch)
    pass
    

    


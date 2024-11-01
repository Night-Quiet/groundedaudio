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


def deep_update(original, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            if len(value) == 0:
                original[key] = value
            deep_update(original[key], value)
        else:
            original[key] = value


def sensevoice_pytorch_save(model_dir, save_dir):
    model_dir = Path(model_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sensevoice, kwargs = SenseVoiceSmall.from_pretrained(model_dir)
    torch.save(sensevoice, save_dir / f"{model_dir.name}.pth")
    torch.save(sensevoice.state_dict(), save_dir / f"{model_dir.name}.pt")
    joblib.dump(kwargs, save_dir / "kwargs.joblib")


def sensevoice_pytorch_load(model_dir):
    model_dir = Path(model_dir)

    # kwargs = joblib.load(model_dir / "kwargs.joblib")
    config = GroundedAudioConfig.from_json_file("./config.json")

    # model_conf = {}
    # deep_update(model_conf, kwargs.get("model_conf", {}))
    # deep_update(model_conf, kwargs)

    # state_dict = torch.load(model_dir / "SenseVoiceSmall.pt")
    
    # pop_keys = [key for key in state_dict.keys() if "ctc" in key]
    # for key in pop_keys:
        # state_dict.pop(key)
    # same effect
    # state_dict.pop("ctc.ctc_lo.weight")
    # state_dict.pop("ctc.ctc_lo.bias")
    
    model = SenseVoiceSmall(config.backbone_config)
    # model.load_state_dict(state_dict)

    return model


def bert_pytorch_save(model_dir, save_dir=None):
    model = BertModel.from_pretrained(model_dir)
    state_dict = model.state_dict()
    pop_keys = [key for key in state_dict.keys() if "pooler.dense" in key]
    for key in pop_keys:
        state_dict.pop(key)
    torch.save(state_dict, "/root/groundedaudio_pretrained/text_encoder.pt")


def json_load(path):
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data


def json_dump(data, path, indent=None):
    with open(path, "w", encoding="utf-8") as f:
        # ensure_ascii=False: 保证输出中文可视化
        json.dump(data, f, ensure_ascii=False, indent=indent)


def audioset_strong_copy():
    audio_dir = "/root/autodl-tmp/audioset"

    for name in ["train", "eval"]:
        csv_data = pd.read_csv(f"./audioset/audioset_{name}_strong.tsv", sep="\t")
        csv_data_grouped = csv_data.groupby("segment_id")
        for segment_id, group in tqdm(csv_data_grouped):
            audio_file_eval_path = os.path.join(audio_dir, "eval", "Y"+segment_id.rsplit("_", 1)[0]+".wav")
            audio_file_balanced_path = os.path.join(audio_dir, "balanced", "Y"+segment_id.rsplit("_", 1)[0]+".wav")
            audio_file_unbalanced_path = os.path.join(audio_dir, "unbalanced", "Y"+segment_id.rsplit("_", 1)[0]+".wav")
            if os.path.exists(audio_file_eval_path):
                audio_file_path = audio_file_eval_path
            elif os.path.exists(audio_file_balanced_path):
                audio_file_path = audio_file_balanced_path
            elif os.path.exists(audio_file_unbalanced_path):
                audio_file_path = audio_file_unbalanced_path
            else:
                audio_file_path = None
            
            if audio_file_path is None:
                continue
            with wave.open(audio_file_path, 'rb') as wav_file:
                n_frames = wav_file.getnframes()
            if n_frames == 0:
                continue
            shutil.copy(audio_file_path, f"/root/autodl-tmp/audioset_strong/{name}/Y{segment_id.rsplit("_", 1)[0]}.wav")


def audioset_csv_process():
    audio_dir = "/root/autodl-tmp/audioset_unzip"
    
    # class_labels_indices = pd.read_csv("class_labels_indices.csv")
    # class2labels = dict()
    # for i in range(len(class_labels_indices)):
    #     class2labels[class_labels_indices.loc[i, "mid"]] = class_labels_indices.loc[i, "display_name"]

    lack_labels = {
        "/m/0174k2": "Washing machine",
        "/m/018p4k": "Cart",
        "/m/01j2bj": "Bathroom sounds",
        "/m/01j3j8": "Studio recording, Music",
        "/m/01lynh": "Stairs",
        "/m/02417f": "Windscreen wiper, windshield wiper",
        "/m/0269r2s": "Chain",
        "/m/02f9f_": "Shower",
        "/m/02ll1_": "Lock",
        "/m/040b_t": "Refrigerator",
        "/m/04ctx": "Knife",
        "/m/056r_1": "Keypress tone",
        "/m/0641k": "Paper rustling",
        "/m/06cyt0": "Mechanical bell",
        "/m/07pqmly": "Slurp, drinking straw",
        "/m/07s13rg": "Sweeping",
        "/m/07sk0jz": "Stomp, stamp",
        "/m/08dckq": "Carbon monoxide detector, CO detector",
        "/m/098_xr": "Error signal",
        "/m/0bcdqg": "Ringing tone, ringback tone",
        "/m/0bzvm2": "Video game sound",
        "/m/0c1tlg": "Electric rotor drone, quadcopter",
        "/m/0d4wf": "Kitchen and dining room sounds",
        "/m/0fw86": "Tap dance",
        "/m/0hgq8df": "Crockery breaking and smashing",
        "/m/0md09": "Power saw, circular saw, table saw",
        "/t/dd00138": "Brief tone",
        "/t/dd00141": "Pant (dog)",
        "/t/dd00142": "Audio logo",
        "/t/dd00143": "Unknown sound",
        "/t/dd00144": "Alert",
        "/t/dd00147": "Dong, bong"
    }

    class_labels_indices = json_load("./audioset/ontology.json")
    class2labels = lack_labels
    for class_labels_indice in class_labels_indices:
        class2labels[class_labels_indice["id"]] = class_labels_indice["name"]

    json_dump(class2labels, "./audioset/class2labels.json", indent=4)
    class2labels = json_load("./audioset/class2labels.json")

    for name in ["train", "eval"]:
        new_audioset_csv = list()

        csv_data = pd.read_csv(f"./audioset/audioset_{name}_strong.tsv", sep="\t")
        csv_data_grouped = csv_data.groupby("segment_id")
        for segment_id, group in tqdm(csv_data_grouped):

            audio_file_eval_path = os.path.join(audio_dir, "eval", "Y"+segment_id.rsplit("_", 1)[0]+".wav")
            audio_file_balanced_path = os.path.join(audio_dir, "balanced", "Y"+segment_id.rsplit("_", 1)[0]+".wav")
            audio_file_unbalanced_path = os.path.join(audio_dir, "unbalanced", "Y"+segment_id.rsplit("_", 1)[0]+".wav")
            if os.path.exists(audio_file_eval_path):
                audio_file_path = audio_file_eval_path
            elif os.path.exists(audio_file_balanced_path):
                audio_file_path = audio_file_balanced_path
            elif os.path.exists(audio_file_unbalanced_path):
                audio_file_path = audio_file_unbalanced_path
            else:
                audio_file_path = None
            
            if audio_file_path is None:
                continue
            with wave.open(audio_file_path, 'rb') as wav_file:
                n_frames = wav_file.getnframes()
            if n_frames == 0:
                continue

            group_temp = dict()
            for i in range(len(group)):
                gi = group.iloc[i].to_dict()
                group_temp["segment_id"] = "Y"+segment_id.rsplit("_", 1)[0]

                group_temp.setdefault("class_labels", list()).append(class2labels[gi["label"]])
                group_temp.setdefault("boxes", list()).append([(gi["end_time_seconds"]+gi["start_time_seconds"]) / 2, (gi["end_time_seconds"]-gi["start_time_seconds"]) / 2])
            new_audioset_csv.append(group_temp)
        json_dump(new_audioset_csv, f"./audioset/audioset_{name}_strong_transform.json", indent=4)


def audioset_del_weak():
    label_json = json_load("./audioset/label.json")
    new_label_json = list()
    for label in tqdm(label_json):
        segment_id = label["segment_id"]
        class_labels = label["class_labels"]
        boxes = label["boxes"]
        audio_path = None
        if os.path.exists(os.path.join("/root/autodl-tmp/audioset_strong_del_weak/train", f"{segment_id}.wav")):
            audio_path = os.path.join("/root/autodl-tmp/audioset_strong_del_weak/train", f"{segment_id}.wav")
            audio, fs = torchaudio.load(os.path.join("/root/autodl-tmp/audioset_strong_del_weak/train", f"{segment_id}.wav"))
        else:
            audio_path = os.path.join("/root/autodl-tmp/audioset_strong_del_weak/val", f"{segment_id}.wav")

        audio, fs = torchaudio.load(audio_path)

        class_labels_tmp, boxes_tmp = list(), list()
        for c, b in zip(class_labels, boxes):
            if ((b[1]*2) - (audio.size(1) / fs))**2 < 0.0001:
                continue
            class_labels_tmp.append(c)
            boxes_tmp.append(b)
        if len(class_labels_tmp) != 0:
            new_label_json.append({"segment_id": segment_id, "class_labels": class_labels_tmp, "boxes": boxes_tmp})
        else:
            os.remove(audio_path)
    print(len(label_json))
    print(len(new_label_json))
    json_dump(new_label_json, "./audioset/label_del_weak.json", indent=4)


def llava_178K_process():
    
    pass


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
    audioset_del_weak()
    pass

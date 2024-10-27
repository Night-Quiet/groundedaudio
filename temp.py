import os
import wave
import json
import torch
import joblib
import shutil
from pathlib import Path
from groundingaudio.sensevoice_model import SenseVoiceSmall
from transformers import BertModel, BertTokenizer
from groundingaudio.configuration_grounding_audio import GroundingAudioConfig

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


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
    config = GroundingAudioConfig.from_json_file("./config.json")

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
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir)

    tokenizer.save_pretrained("/root/groundingaudio/sensevoice_pytorch")
    model.save_pretrained("/root/groundingaudio/sensevoice_pytorch")


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



if __name__ == "__main__":
    # sensevoice_pytorch_save("/root/groundingaudio/SenseVoiceSmall", "/root/groundingaudio/sensevoice_pytorch")
    # sensevoice_pytorch_load("/root/groundingaudio/sensevoice_pytorch")
    # bert_pytorch_save("/root/autodl-tmp/bert-base-uncased")
    # audioset_csv_process()
    audioset_strong_copy()
    pass




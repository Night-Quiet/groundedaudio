import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import json
import numpy as np
from time import localtime, time
from datasets import load_dataset
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from utils import AudioSetSLPreprocessor
from groundingaudio.processing_grounding_audio import GroundingAudioProcessor
from groundingaudio.grounding_audio_model import GroundingAudioForObjectDetection
from groundingaudio.configuration_grounding_audio import GroundingAudioConfig


class DataCollatorWithPadding:
    def __init__(self, processor, json_file):
        self.processor = processor

        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        segment_id2data = dict()
        for data_i in json_data:
            segment_id2data[data_i["segment_id"]] = {"class_labels": data_i["class_labels"], "boxes": data_i["boxes"]}
        self.json_data = segment_id2data
    
    def labels2ids(self, class_labels, boxes):
        class_labels_ids = list()
        boxes_labels_ids = list()

        sentence_ids = list()
        label2ids = dict()
        
        for ids, label in enumerate(class_labels):
            label2ids.setdefault(label, list()).append(ids)
        
        for label in set(class_labels):
            start_ids = len(sentence_ids) + 1
            label_ids = self.processor.tokenizer.encode(label+".", add_special_tokens=False)
            sentence_ids.extend(label_ids)

            for ids in label2ids[label]:
                for i in range(len(label_ids)-1):
                    class_labels_ids.append(start_ids+i)
                    boxes_labels_ids.append(boxes[ids])

        sentence_ids.pop()
        sentence = self.processor.tokenizer.decode(sentence_ids)

        return sentence, class_labels_ids, boxes_labels_ids
    
    def __call__(self, features):
        import torch

        text_list, audio_list, labels_list = [], [], []
        for i in range(len(features)):
            segment_id = features[i]["audio"]["path"].rsplit("/", 1)[-1].split(".", 1)[0]
            sentence, class_labels_ids, boxes_labels_ids = self.labels2ids(self.json_data[segment_id]["class_labels"], self.json_data[segment_id]["boxes"])
            audio_list.append(features[i]["audio"]["array"])
            text_list.append(sentence)
            
            labels_list.append({
                "class_labels": torch.tensor(class_labels_ids, dtype=torch.long),
                "boxes": torch.tensor(boxes_labels_ids)
            })
        batch = self.processor(audios=audio_list, text=text_list)
        batch["labels"] = labels_list

        return batch


class HyperParameters():
    def __init__(self) -> None:
        # paths
        self.checkpoint_dir = "/root/groundingaudio_pretrained"
        self.data_json_path = "/root/groundingaudio/audioset/audioset_eval_strong_transform.json"
        self.data_audio_dir = "/root/autodl-tmp/audioset_strong/eval"
        self.output_dir = '/root/autodl-tmp/results'
        # train
        self.start_epoch = 0
        self.num_train_epochs = 30
        self.per_device_train_batch_size = 4
        self.per_device_eval_batch_size = 4
        self.eval_strategy="epoch"
        self.save_strategy="epoch"
        self.dataloader_num_workers=10
        self.seed=100
        self.neftune_noise_alpha=None
        # log
        self.report_to="none"
        self.logging_steps=100
        self.run_name="groudingaudio_origin"
        # optimizer
        self.learning_rate = 1e-5
        self.weight_decay = 0.0
        self.warmup_ratio=0.1
        self.lr_scheduler_type="cosine"
        self.betas = (0.9, 0.999)


# config and make results directories
cfg = HyperParameters()
local_time = localtime(time())
time_info = '' if (cfg.start_epoch != 0) else '{}-{}-{}-{}'.format(local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min)
cfg.output_dir = os.path.join(cfg.output_dir, time_info)
os.makedirs(os.path.join(cfg.output_dir, 'log'), exist_ok=True)
json_string = json.dumps(cfg.__dict__, indent=4)
with open(os.path.join(cfg.output_dir, "hyperparameters.json"), 'w') as file:
    file.write(json_string)

# model
config = GroundingAudioConfig.from_json_file(os.path.join(cfg.checkpoint_dir, "config.json"))
model = GroundingAudioForObjectDetection(config)
model.model.freeze_backbone()

# optimizer
params = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=cfg.betas)

# dataset
dataset = load_dataset("audiofolder", data_dir=cfg.data_audio_dir, drop_labels=True, split="train", keep_in_memory=False, cache_dir="/root/autodl-tmp/.cache")
processor = GroundingAudioProcessor.from_pretrained(cfg.checkpoint_dir)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
data_collator = DataCollatorWithPadding(processor=processor, json_file=cfg.data_json_path)

# trainer and train
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        remove_unused_columns=False
    ),
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=processor,
    data_collator=data_collator,
    optimizers=(optimizer, None)
)
trainer.train()
trainer.save_state()

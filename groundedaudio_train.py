import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import json
from time import localtime, time
from datasets import load_dataset
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from groundedaudio.processing_grounded_audio import GroundedAudioProcessor
from groundedaudio.grounded_audio_model import GroundedAudioForObjectDetection
from groundedaudio.configuration_grounded_audio import GroundedAudioConfig
from utils import AudioSetSLPreprocessor


from transformers import DataCollatorWithPadding
import torch

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        text_features = [{
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"],
            "token_type_ids": feature["token_type_ids"]
        } for feature in features]
        
        collated_text_features = self.data_collator(text_features)

        audio_values = [feature["audio_values"] for feature in features]
        audio_mask = [feature["audio_mask"] for feature in features]

        labels_tensor = []
        for feature in features:
            class_labels = torch.tensor(feature["labels"]["class_labels"], dtype=torch.long)
            boxes = torch.tensor(feature["labels"]["boxes"], dtype=torch.float32)
            labels_tensor.append({"class_labels": class_labels, "boxes": boxes})

        max_time_length = max(len(values) for values in audio_values)
        padded_audio_values = []
        for values in audio_values:
            padded_time = values + [[0] * len(values[0])] * (max_time_length - len(values))
            padded_audio_values.append(padded_time)

        padded_audio_mask = [mask + [0] * (max_time_length - len(mask)) for mask in audio_mask]

        # 将所有特征转换为张量
        collated_features = {
            "input_ids": collated_text_features["input_ids"],
            "attention_mask": collated_text_features["attention_mask"],
            "token_type_ids": collated_text_features["token_type_ids"],
            "audio_values": torch.tensor(padded_audio_values, dtype=torch.float32),
            "audio_mask": torch.tensor(padded_audio_mask, dtype=torch.long),
            "labels": labels_tensor
        }

        return collated_features

            
class HyperParameters():
    def __init__(self) -> None:
        # paths
        self.checkpoint_dir = "/root/groundedaudio_pretrained"
        self.data_json_path = "/root/groundedaudio/audioset/audioset_train_strong_transform.json"
        self.data_audio_dir = "/root/autodl-tmp/audioset_strong/train"
        self.output_dir = '/root/autodl-tmp/gaudio'
        self.cache_dir = "/root/autodl-tmp/.cache"
        # train
        self.start_epoch = 0
        self.num_train_epochs = 30
        self.per_device_train_batch_size = 64
        self.per_device_eval_batch_size = 64
        self.eval_strategy="epoch"
        self.save_strategy="epoch"
        self.dataloader_num_workers=10
        self.seed=100
        self.neftune_noise_alpha=None
        # log
        # self.report_to="tensorboard"
        self.logging_steps=100
        self.run_name="gaudio_origin"
        # optimizer
        self.learning_rate = 1e-5
        self.weight_decay = 0.0
        self.warmup_ratio=0.1
        self.lr_scheduler_type="cosine"
        self.betas = (0.9, 0.999)
        # models
        self.sensevoice_layer = 50


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
config = GroundedAudioConfig.from_json_file(os.path.join(cfg.checkpoint_dir, "config.json"))
config.backbone_layer = cfg.sensevoice_layer
model = GroundedAudioForObjectDetection(config)
model.model.freeze_backbone()

# optimizer
params = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=cfg.betas)

# dataset
dataset = load_dataset("audiofolder", data_dir=cfg.data_audio_dir, drop_labels=True, keep_in_memory=False, split="train", cache_dir=cfg.cache_dir)
processor = GroundedAudioProcessor.from_pretrained(cfg.checkpoint_dir)
preprocessor = AudioSetSLPreprocessor(processor=processor, json_file=cfg.data_json_path)
dataset = dataset.map(preprocessor, batched=True, remove_columns=["audio"])
dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
data_collator = CustomDataCollator(processor.tokenizer)

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
        bf16=True,
        bf16_full_eval=True,
        remove_unused_columns=False,
        do_train=True
    ),
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=processor,
    data_collator=data_collator,
    optimizers=(optimizer, None)
)
trainer.train()
trainer.save_state()

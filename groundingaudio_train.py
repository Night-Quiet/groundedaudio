import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import json
from time import localtime, time
from datasets import load_dataset
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from utils import AudioSetSLPreprocessor
from groundingaudio.processing_grounding_audio import GroundingAudioProcessor
from groundingaudio.grounding_audio_model import GroundingAudioForObjectDetection
from groundingaudio.configuration_grounding_audio import GroundingAudioConfig


class HyperParameters():
    def __init__(self) -> None:
        # paths
        self.checkpoint_dir = "/root/groundingaudio_pretrained"
        self.data_json_path = "/root/groundingaudio/audioset/audioset_train_strong_transform.json"
        self.data_audio_dir = "/root/autodl-tmp/audioset_strong/train"
        self.output_dir = '/root/autodl-tmp/results'
        # train
        self.start_epoch = 0
        self.train_epochs = 30
        self.per_device_train_batch_size = 8
        self.per_device_eval_batch_size = 8
        self.evaluation_strategy="epoch",
        self.save_strategy="epoch",
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

# dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = load_dataset("json", data_files=cfg.data_json_path, split="train", keep_in_memory=True)
processor = GroundingAudioProcessor.from_pretrained(cfg.checkpoint_dir)
preprocessor = AudioSetSLPreprocessor(processor=processor, audio_dir=cfg.data_audio_dir, device=device)
dataset = dataset.map(preprocessor, batched=True, remove_columns=["segment_id", "class_labels", "boxes"], batch_size=1000).train_test_split(test_size=0.2, shuffle=True)

# trainer and train
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.train_epochs,
        weight_decay=cfg.weight_decay,
        eval_strategy=cfg.evaluation_strategy,
        save_strategy=cfg.save_strategy
    ),
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=processor,
)
trainer.train()
trainer.save_state()

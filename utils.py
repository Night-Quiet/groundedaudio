import os
import torch
from datasets import load_dataset


class AudioSetSLPreprocessor():
    def __init__(self, processor, audio_dir, device):
        self.processor = processor
        self.audio_dir = audio_dir
        self.device = device

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

    def __call__(self, examples):
        text_list, audio_list, labels_list = [], [], []
        for i in range(len(examples["segment_id"])):
            sentence, class_labels_ids, boxes_labels_ids = self.labels2ids(examples["class_labels"][i], examples["boxes"][i])
            audio_file_path = os.path.join(self.audio_dir, examples["segment_id"][i]+".wav")

            audio_list.append(audio_file_path)
            text_list.append(sentence)
            
            labels_list.append({
                "class_labels": torch.tensor(class_labels_ids, device=self.device, dtype=torch.long),
                "boxes": torch.tensor(boxes_labels_ids, device=self.device)
            })
        inputs = self.processor(audios=audio_list, text=text_list, device=self.device)
        inputs["labels"] = labels_list

        return inputs


if __name__ == "__main__":
    dataset = load_dataset("json", data_files="./audioset/audioset_train_strong_transform.json")
    print(dataset["train"][0])
    pass

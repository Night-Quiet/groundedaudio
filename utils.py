import os
import json
import torch
from datasets import load_dataset


class AudioSetSLPreprocessor():
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

    def __call__(self, examples):
        text_list, audio_list, labels_list = [], [], []
        for i in range(len(examples["audio"])):
            segment_id = examples["audio"][i]["path"].rsplit("/", 1)[-1].split(".", 1)[0]
            sentence, class_labels_ids, boxes_labels_ids = self.labels2ids(self.json_data[segment_id]["class_labels"], self.json_data[segment_id]["boxes"])
            audio_list.append(examples["audio"][i]["array"])
            text_list.append(sentence)
            
            labels_list.append({
                "class_labels": torch.tensor(class_labels_ids, dtype=torch.long),
                "boxes": torch.tensor(boxes_labels_ids)
            })
        inputs = self.processor(audios=audio_list, text=text_list)
        inputs["labels"] = labels_list

        return inputs


if __name__ == "__main__":
    dataset = load_dataset("json", data_files="./audioset/audioset_train_strong_transform.json")
    print(dataset["train"][0])
    pass

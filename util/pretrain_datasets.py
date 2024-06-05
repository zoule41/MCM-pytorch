from copy import deepcopy
import os
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import tokenizers
from transformers import BertTokenizer
import random


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MultimodalBertDataset(Dataset):
    def __init__(
            self,
            csv_file,
            transform,
            max_caption_length: int = 100,
            max_words=112
    ):
        self.csv_file = csv_file
        self.max_caption_length = max_caption_length
        self.transform = transform

        self.images_list, self.report_list, self.complaint = self.read_csv()
        self.tokenizer = tokenizers.Tokenizer.from_file("./mimic_wordpiece.json")
        self.tokenizer2 = BertTokenizer.from_pretrained(
            "/root/autodl-tmp/Bio_ClinicalBERT")
        self.max_words = max_words
        self.idxtoword = {v: k for k, v in
                          self.tokenizer.get_vocab().items()}

    def __len__(self):
        return len(self.images_list)

    def _random_mask(self, tokens):
        masked_tokens = deepcopy(tokens)
        for i in range(1, masked_tokens.shape[1] - 1):
            if masked_tokens[0][i] == 0:
                break

            if masked_tokens[0][i - 1] == 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                masked_tokens[0][i] = 3
                continue

            if masked_tokens[0][i - 1] != 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                continue

            prob = random.random()
            if prob < 0.5:
                masked_tokens[0][i] = 3

        return masked_tokens

    def __getitem__(self, index):
        image = pil_loader(self.images_list[index])
        image = self.transform(image)
        sent = self.report_list[index]
        sent = '[CLS] ' + sent
        complaint = self.complaint[index]
        complaint_tokens = self.tokenizer2(
            complaint,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )

        self.tokenizer.enable_truncation(max_length=self.max_caption_length)
        self.tokenizer.enable_padding(length=self.max_caption_length)

        encoded = self.tokenizer.encode(sent)
        ids = torch.tensor(encoded.ids).unsqueeze(0)
        attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(
            0)
        type_ids = torch.tensor(encoded.type_ids).unsqueeze(0)

        ids_2 = complaint_tokens["input_ids"]
        attention_mask_2 = complaint_tokens["attention_mask"]
        type_ids_2 = complaint_tokens["token_type_ids"]

        masked_ids = self._random_mask(ids)
        return image, ids, attention_mask, type_ids, masked_ids, ids_2, attention_mask_2, type_ids_2

    def read_csv(self):
        csv_path = self.csv_file
        df = pd.read_csv(csv_path, sep=',')
        return df["Path"], df["cleaned_report"], df["cleaned_complaint"]

    def collate_fn(self, instances: List[Tuple]):
        (image_list, ids_list, attention_mask_list, type_ids_list, masked_ids_list, ids_2_list,
         attention_mask_2_list, type_ids_2_list) = [], [], [], [], [], [], [], []
        # flattern
        for b in instances:
            image, ids, attention_mask, type_ids, masked_ids, ids_2, attention_mask_2, type_ids_2 = b
            image_list.append(image)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)
            type_ids_list.append(type_ids)
            masked_ids_list.append(masked_ids)
            ids_2_list.append(ids_2)
            attention_mask_2_list.append(attention_mask_2)
            type_ids_2_list.append(type_ids_2)

        # stack
        image_stack = torch.stack(image_list)
        ids_stack = torch.stack(ids_list).squeeze()
        attention_mask_stack = torch.stack(attention_mask_list).squeeze()
        type_ids_stack = torch.stack(type_ids_list).squeeze()
        masked_ids_stack = torch.stack(masked_ids_list).squeeze()
        ids_2_stack = torch.stack(ids_2_list).squeeze()
        attention_mask_2_stack = torch.stack(attention_mask_2_list).squeeze()
        type_ids_2_stack = torch.stack(type_ids_2_list).squeeze()

        # sort and add to dictionary
        return_dict = {
            "image": image_stack,
            "labels": ids_stack,
            "attention_mask": attention_mask_stack,
            "type_ids": type_ids_stack,
            "ids": masked_ids_stack,
            "ids_2": ids_2_stack,
            "attention_mask_2": attention_mask_2_stack,
            "type_ids_2": type_ids_2_stack
        }

        return return_dict

import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


class LanguageDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def build_language_dataloaders(jsonl_path, tokenizer, batch_size=32, max_length=128):
    lan_to_texts = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            lan_to_texts[item["lan"]].append(item["text"])

    def collate_fn(batch):
        return tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    dataloaders = {}
    for lan, texts in lan_to_texts.items():
        dataset = LanguageDataset(texts)
        dataloaders[lan] = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn
        )

    return dataloaders


class CodeSwitchDataset(Dataset):
    def __init__(self, jsonl_path, target_lan):
        self.data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                if item["target_lan"] == target_lan:
                    self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_code_switch_collate_fn(tokenizer):
    def collate_fn(batch):
        sentences = [item["sentence"] for item in batch]
        prefixes = [item["ori_sentence"] for item in batch]

        full_encodings = tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        )
        prefix_encodings = tokenizer(
            prefixes, return_tensors="pt", padding=True, truncation=True
        )

        batch_size = full_encodings.input_ids.shape[0]
        seq_len = full_encodings.input_ids.shape[1]

        # Create a boolean mask to isolate the "noun" tokens (tokens after the prefix)
        noun_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

        # We also need the isolated noun (BOS token + Noun)
        isolated_nouns = []

        for i in range(batch_size):
            p_len = prefix_encodings.attention_mask[i].sum().item()
            f_len = full_encodings.attention_mask[i].sum().item()

            # Mask exactly where the noun is in the padded full sequence
            noun_mask[i, p_len:f_len] = True

            # Extract just the noun tokens and prepend the BOS token (if applicable)
            bos = full_encodings.input_ids[i, 0].unsqueeze(0)
            noun_tokens = full_encodings.input_ids[i, p_len:f_len]
            isolated_nouns.append(torch.cat([bos, noun_tokens]))

        # Pad the isolated nouns for the second forward pass
        isolated_encodings = tokenizer.pad(
            {"input_ids": isolated_nouns}, return_tensors="pt", padding=True
        )

        return {
            "full_input_ids": full_encodings.input_ids,
            "full_attention_mask": full_encodings.attention_mask,
            "noun_mask": noun_mask,
            "isolated_input_ids": isolated_encodings.input_ids,
            "isolated_attention_mask": isolated_encodings.attention_mask,
        }

    return collate_fn

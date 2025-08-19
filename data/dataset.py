from datasets import load_dataset
import torch

def load_snli_dataset(split="train"):
    dataset = load_dataset("snli", split=split, download_mode="reuse_cache_if_exists")
    dataset = dataset.filter(lambda x: x['label'] != -1)
    return [(x['premise'], x['hypothesis'], x['label']) for x in dataset]

def load_multinli_dataset(split="train"):
    dataset = load_dataset("multi_nli", split=split, download_mode="reuse_cache_if_exists")
    dataset = dataset.filter(lambda x: x['label'] != -1)
    return [(x['premise'], x['hypothesis'], x['label']) for x in dataset]

def load_combined_nli_dataset():
    snli = load_snli_dataset("train")
    multinli = load_multinli_dataset("train")
    return snli + multinli

def collate_fn(batch, tokenizer):
    sentence1 = [item[0] for item in batch]
    sentence2 = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    inputs1 = tokenizer(sentence1, padding=True, truncation=True, return_tensors="pt")
    inputs2 = tokenizer(sentence2, padding=True, truncation=True, return_tensors="pt")

    return {
        'input_ids1': inputs1['input_ids'],
        'attention_mask1': inputs1['attention_mask'],
        'input_ids2': inputs2['input_ids'],
        'attention_mask2': inputs2['attention_mask'],
        'labels': labels
    }

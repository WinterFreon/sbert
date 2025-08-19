import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from datasets import load_dataset
from transformers import logging
from model.sentence_bert import SentenceBERT

logging.set_verbosity_error()


def evaluate_cosine_spearman(model, tokenizer, sents1, sents2, scores, device):
    model.eval()
    batch_size = 32
    all_embeddings1, all_embeddings2 = [], []

    with torch.no_grad():
        for i in range(0, len(sents1), batch_size):
            batch1 = sents1[i:i + batch_size]
            batch2 = sents2[i:i + batch_size]

            tokens1 = tokenizer(batch1, padding=True, truncation=True, return_tensors='pt', max_length=128)
            tokens2 = tokenizer(batch2, padding=True, truncation=True, return_tensors='pt', max_length=128)
            tokens1, tokens2 = {k: v.to(device) for k, v in tokens1.items()}, {k: v.to(device) for k, v in tokens2.items()}

            emb1 = model(tokens1['input_ids'], tokens1['attention_mask'])
            emb2 = model(tokens2['input_ids'], tokens2['attention_mask'])

            all_embeddings1.append(emb1.cpu())
            all_embeddings2.append(emb2.cpu())

    emb1, emb2 = torch.cat(all_embeddings1, dim=0), torch.cat(all_embeddings2, dim=0)
    cosine_scores = torch.nn.functional.cosine_similarity(emb1, emb2).numpy()
    gold_scores = np.array(scores) / 5.0

    spearman_corr, _ = spearmanr(cosine_scores, gold_scores)
    return spearman_corr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks = {
        "STS12": ("mteb/sts12-sts", "test"),
        "STS13": ("mteb/sts13-sts", "test"),
        "STS-B": ("glue", "validation")  # GLUE test set restricted
    }

    poolings = ["mean", "max", "cls", "mean_max"]
    all_results = {}

    for pooling in poolings:
        print(f"\n===== Evaluating pooling: {pooling} =====")
        model = SentenceBERT(pooling=pooling).to(device)
        model.load_state_dict(torch.load(f"sbert_regression_{pooling}.pth", map_location=device))
        model.eval()
        tokenizer = model.tokenizer

        results = {}
        for name, (dataset_name, split) in tasks.items():
            try:
                print(f"Task: {name}")
                if name == "STS-B":
                    dataset = load_dataset(dataset_name, "stsb", split=split)
                    sents1, sents2, scores = dataset["sentence1"], dataset["sentence2"], dataset["label"]
                else:
                    dataset = load_dataset(dataset_name, split=split)
                    sents1, sents2, scores = dataset["sentence1"], dataset["sentence2"], dataset["score"]

                spearman = evaluate_cosine_spearman(model, tokenizer, sents1, sents2, scores, device)
                results[name] = spearman
                print(f" Spearman: {spearman:.4f}")
            except Exception as e:
                print(f" Failed {name}: {e}")

        avg = np.mean(list(results.values())) if results else 0.0
        results["Average"] = avg
        all_results[pooling] = results
        print(f"==> {pooling} Average Spearman: {avg:.4f}")

    # save
    with open("sts_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n✅ Saved results to sts_results.json")


if __name__ == "__main__":
    main()

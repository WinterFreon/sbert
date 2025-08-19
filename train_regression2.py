import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model.sentence_bert import SentenceBERT
from model.objectives import RegressionObjective
from tqdm import tqdm
from datasets import load_dataset
from scipy.stats import spearmanr
import wandb


def load_sts_dataset(split="train"):
    dataset = load_dataset("glue", "stsb", split=split)
    sentence1 = dataset["sentence1"]
    sentence2 = dataset["sentence2"]
    scores = dataset["label"]
    return list(zip(sentence1, sentence2, scores))


def collate_fn(batch, tokenizer):
    sents1 = [item[0] for item in batch]
    sents2 = [item[1] for item in batch]
    labels = torch.tensor([item[2] / 5.0 for item in batch], dtype=torch.float32)  # Normalize to [0,1]

    inputs1 = tokenizer(sents1, padding=True, truncation=True, return_tensors="pt")
    inputs2 = tokenizer(sents2, padding=True, truncation=True, return_tensors="pt")

    return {
        "input_ids1": inputs1["input_ids"],
        "attention_mask1": inputs1["attention_mask"],
        "input_ids2": inputs2["input_ids"],
        "attention_mask2": inputs2["attention_mask"],
        "labels": labels
    }


def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    gold_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            u = model(batch["input_ids1"], batch["attention_mask1"])
            v = model(batch["input_ids2"], batch["attention_mask2"])

            cos_sim = torch.cosine_similarity(u, v)
            predictions.extend(cos_sim.cpu().numpy())
            gold_scores.extend(batch["labels"].cpu().numpy())

    spearman_corr = spearmanr(predictions, gold_scores).correlation
    return spearman_corr


def main():
    wandb.init(project="sentence-bert-project", name="ep1-regression-finetune-train-regression2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceBERT().to(device)
    tokenizer = model.tokenizer
    model.load_state_dict(torch.load("my_best_sbert_model.pth", map_location=device))

    # ✅ print weight std before train
    print("🔍 Before training - encoder layer[0] weight std:", model.bert.encoder.layer[0].intermediate.dense.weight.std().item())

    train_data = load_sts_dataset("train")
    val_data = load_sts_dataset("validation")

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer))

    objective = RegressionObjective().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(1):
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            u = model(batch["input_ids1"], batch["attention_mask1"])
            v = model(batch["input_ids2"], batch["attention_mask2"])

            loss = objective(u, v, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        spearman = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} Loss: {avg_train_loss:.4f}")
        print(f"Validation Spearman Correlation: {spearman:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_spearman": spearman
        })

    # ✅ save encoder-only model（for STS similarity assessment）
    torch.save(model.state_dict(), "sbert_regression_encoder_only.pth")
    print("✅ Encoder-only model saved as sbert_regression_encoder_only.pth")

    # ✅ print weight again std（make sure whether update）
    print("🔍 After training - encoder layer[0] weight std:", model.bert.encoder.layer[0].intermediate.dense.weight.std().item())

    wandb.finish()


if __name__ == "__main__":
    main()

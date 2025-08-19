import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import wandb
import argparse

from model.sentence_bert import SentenceBERT
from model.objectives import ClassificationObjective
from data.dataset import load_combined_nli_dataset, load_multinli_dataset, collate_fn
from config import config

parser = argparse.ArgumentParser()
parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls", "mean_max"])
args = parser.parse_args()
config['pooling'] = args.pooling

def evaluate(model, classifier, dataloader, device):
    model.eval()
    classifier.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            u = model(batch['input_ids1'], batch['attention_mask1'])
            v = model(batch['input_ids2'], batch['attention_mask2'])
            features = torch.cat([u, v, torch.abs(u - v)], dim=1)
            logits = classifier.classifier(features)
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)

    return correct_predictions / total_samples

def main():
    wandb.init(project="sentence-bert-project", name=f"epoch-{config['num_epochs']}-train-{config['pooling']}", config=config)

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    model = SentenceBERT(config['model_name'], pooling=config['pooling']).to(device)

    ##
    if args.pooling == "mean_max":
        embedding_dim = 768 * 2
    else:
        embedding_dim = 768

    classifier = ClassificationObjective(embedding_dim, config['num_labels']).to(device)

    #
    train_data = load_combined_nli_dataset()
    val_data = load_multinli_dataset("validation_matched")
    tokenizer = model.tokenizer

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer))

    total_steps = len(train_loader) * config['num_epochs']
    warmup_steps = int(0.1 * total_steps)
    optimizer = AdamW(list(model.parameters()) + list(classifier.parameters()), lr=config['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_accuracy = 0.0
    for epoch in range(config['num_epochs']):
        model.train()
        classifier.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            u = model(batch['input_ids1'], batch['attention_mask1'])
            v = model(batch['input_ids2'], batch['attention_mask2'])
            loss = classifier(u, v, batch['labels'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        val_accuracy = evaluate(model, classifier, val_loader, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch+1} Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f"my_best_sbert_model_{config['pooling']}.pth")
            print(f"New best model saved: my_best_sbert_model_{config['pooling']}.pth")

    wandb.finish()

if __name__ == '__main__':
    main()

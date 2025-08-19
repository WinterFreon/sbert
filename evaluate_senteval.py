import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import logging

from model.sentence_bert import SentenceBERT

logging.set_verbosity_error()

def get_sentence_embeddings(model, tokenizer, sentences, device, batch_size=32):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            emb = model(encoded['input_ids'], encoded['attention_mask'])
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0).numpy()

def evaluate_classification_task(model, tokenizer, task_name, dataset_name, split_name, sent1_field, sent2_field=None, label_field="label"):
    print(f"\nEvaluating {task_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    if isinstance(dataset_name, tuple):
        dataset = load_dataset(*dataset_name, split=split_name, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=split_name, trust_remote_code=True)

    if sent2_field:
        sentences = [s1 + " [SEP] " + s2 for s1, s2 in zip(dataset[sent1_field], dataset[sent2_field])]
    else:
        sentences = dataset[sent1_field]

    labels = dataset[label_field]

    # Split manually into train/test if needed
    n = len(labels)
    split = int(0.8 * n)
    train_sentences, test_sentences = sentences[:split], sentences[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    # Get embeddings
    train_embeddings = get_sentence_embeddings(model, tokenizer, train_sentences, device)
    test_embeddings = get_sentence_embeddings(model, tokenizer, test_sentences, device)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_embeddings, train_labels)

    # Predict
    preds = clf.predict(test_embeddings)
    acc = accuracy_score(test_labels, preds)
    print(f"{task_name} Accuracy: {acc:.4f}")
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceBERT().to(device)
    model.load_state_dict(torch.load("sbert_regression_encoder_only.pth", map_location=device))
    model.eval()
    tokenizer = model.tokenizer

    tasks = [
        ("SUBJ", "SetFit/subj", "train", "text", None, "label"),
        ("SST-2", ("glue", "sst2"), "train", "sentence", None, "label"),
        ("MRPC", ("glue", "mrpc"), "train", "sentence1", "sentence2", "label"),
    ]

    results = {}
    for task in tasks:
        acc = evaluate_classification_task(model, tokenizer, *task)
        results[task[0]] = acc

    print("\n==== Overall Results ====")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
    avg = np.mean(list(results.values()))
    print(f"\nAverage Accuracy: {avg:.4f}")

if __name__ == "__main__":
    main()

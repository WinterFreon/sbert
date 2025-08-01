# 🧠 Sentence-BERT Replication (SBERT)

This project replicates the experiments from the paper:  
**"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"**  
[(Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084)

---

## 📌 Overview

We reproduce the **Sentence-BERT** architecture using PyTorch and HuggingFace Transformers.  
Our goal is to fine-tune BERT on NLI datasets and evaluate sentence embeddings on:
- **STS Benchmarks** (Semantic Textual Similarity)
- **SentEval Transfer Tasks** (TREC, MRPC, etc.)

---

## 📁 Project Structure

- **model/**
  - `sentence_bert.py` — base architecture
  - `objectives.py`
- **data/**
  - `dataset.py` 
- `train.py` — Train with classification objective
- `train_regression.py` — Train with cosine regression objective
- `train_regression2.py` — Save encoder-only model
- `evaluate_all_sts.py` — Evaluate on STS Benchmark
- `evaluate_senteval.py` — Evaluate on SentEval classification tasks
- `config.py` 
- `requirements.txt` 
- `README.md` 

---

## ⚙️ Environment Setup

### 🔧 Requirements

- Python >= 3.8
- PyTorch >= 1.10
- Transformers >= 4.30
- Datasets
- scikit-learn
- scipy
- wandb (optional)

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

- 0️⃣ Step 0: Build from 'bert-base-uncased'
```bash
- `sentence_bert.py`
- `objectives.py`
```
- 1️⃣ Step 1: Train Model with SNLI + MultiNLI
```bash
- `train.py`
```
- 2️⃣ Step 2: Train SBERT with Regression Loss
```bash
- `train_regression.py`
```
This fine-tunes BERT on SNLI + MultiNLI using cosine-similarity regression loss.
The model is saved as: `sbert_regression_finetuned.pth`
- 3️⃣ Step 3: Evaluate on STS Benchmark
```bash
- `evaluate_all_sts.py`
```
Sample output:
```bash
Spearman correlation on STS Benchmark (test): 0.8134
```
- 4️⃣ Step 4: Evaluate on SentEval Transfer Tasks
```bash
- evaluate_senteval.py
```
---
## 🧪 Reproducibility
✅ All datasets are downloaded via datasets library and cached

✅ Configurable batch size, epochs, learning rate via `config.py`

✅ Model training is logged via optional `wandb` support

✅ All evaluation scripts rely on saved `.pth` models for reproducibility

---
## 📊 Experimental Results
| Task           | Metric   | Result |
| -------------- | -------- | ------ |
| STS12          | Spearman | 0.7181 |
| STS13          | Spearman | 0.8081 |
| STS-B          | Spearman | 0.8134 |
| SentEval SUBJ  | Accuracy | 0.9131 |
| SentEval SST-2 | Accuracy | 0.8696 |
| SentEval MRPC  | Accuracy | 0.7139 |

---
## 🙌 Acknowledgements
This project replicates the work of:
Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
by Nils Reimers and Iryna Gurevych

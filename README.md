# SBERT Reproduction and Extension

This project replicates the experiments from the paper:  
**"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"**  
[(Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084)
and **extends it** with new experiments on pooling strategies and training dynamics.

---
## 💡 What’s New

- Added **MEAN+MAX pooling** (not in the original paper).  
- Systematic comparison of **four pooling methods**: MEAN, MAX, CLS, MEAN+MAX.  
- Evaluation on **STS12, STS13, STS-B** and **SentEval** (SUBJ, SST-2, MRPC).  
- **Extended training dynamics experiment**: multi-epoch monitoring of loss/accuracy (original paper only used 1 epoch).  
- Results stored in JSON (`sts_results.json`, `senteval_results.json`) and visualized in bar plots.

---

## 📁 Project Structure
### sbert-main
- **model/**
  - `sentence_bert.py` — base architecture
  - `objectives.py`
- **data/**
  - `dataset.py` 
- `train.py` 
- `train_regression.py` — Save encoder-only model
- `evaluate_all_sts.py` — Evaluate on STS Benchmark
- `evaluate_senteval.py` — Evaluate on SentEval Evaluation
- `config.py` 
- `requirements.txt`
  
### sbert-pooling
- **model/**
  - `sentence_bert.py` 
  - `objectives.py`
- **data/**
  - `dataset.py` 
- `train.py`
- `train_regression.py`
- `train_regression_all_pooling.py`
- `evaluate_sts_.py` 
- `evaluate_senteval_.py` 
- `config.py` 

---

## ⚙️ Environment Setup

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- Transformers >= 4.30
- Datasets
- scikit-learn
- scipy
- wandb (optional)

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📘 Steps
### sbert-main
- Step 0: Build from 'bert-base-uncased'
```bash
- `sentence_bert.py`
- `objectives.py`
```
- Step 1: Train Model with SNLI + MultiNLI
```bash
- `train.py`
```
- Step 2: Train SBERT with Regression Loss
```bash
- `train_regression.py`
```
This fine-tunes BERT on SNLI + MultiNLI using cosine-similarity regression loss.
The model is saved as: `sbert_regression_finetuned.pth`
- Step 3: Evaluate on STS Benchmark
```bash
- `evaluate_all_sts.py`
```
Sample output:
```bash
Spearman correlation on STS Benchmark (test): 0.8134
```
- Step 4: Evaluate on SentEval Transfer Tasks
```bash
- `evaluate_senteval.py`
```
### sbert-pooling
- Step 0:  Train NLI model
```bash
- !python train.py --pooling mean #max,cls,mean+max
```
This produces: my_best_sbert_model_{pooling}.pth
- Step 1: Fine-tune on STS-B regression
```bash
- !python train_regression.py
```
This produces: sbert_regression_{pooling}.pth
- Step 2: Evaluate
```bash
- !python evaluate_sts_.py
- !python evaluate_senteval_.py
```
Results are automatically saved to JSON (sts_results.json, senteval_results.json) and plotted.

---

## 📊 Experimental Results
### sentence-bert-project
| Task           | Metric   | Result |
| -------------- | -------- | ------ |
| STS12          | Spearman | 0.7181 |
| STS13          | Spearman | 0.8081 |
| STS-B          | Spearman | 0.8134 |
| SentEval SUBJ  | Accuracy | 0.9131 |
| SentEval SST-2 | Accuracy | 0.8696 |
| SentEval MRPC  | Accuracy | 0.7139 |
### sbert-pooling
STS12/STS13/STS-B (Spearman ρ)
| Pooling  | STS12  | STS13  | STS-B  | Avg    |
| -------- | ------ | ------ | ------ | ------ |
| MEAN     | 0.7795 | 0.8705 | 0.8587 | 0.8362 |
| CLS      | 0.7604 | 0.8670 | 0.8563 | 0.8279 |
| MEAN+MAX | 0.7683 | 0.8199 | 0.8272 | 0.8052 |
| MAX      | 0.7436 | 0.7546 | 0.7764 | 0.7582 |

SentEval (Accuracy)
| Pooling  | SUBJ   | SST-2  | MRPC   | Avg    |
| -------- | ------ | ------ | ------ | ------ |
| MEAN     | 0.9150 | 0.8807 | 0.6948 | 0.8302 |
| CLS      | 0.9175 | 0.8791 | 0.6621 | 0.8196 |
| MEAN+MAX | 0.8963 | 0.8843 | 0.6499 | 0.8101 |
| MAX      | 0.8900 | 0.8675 | 0.6703 | 0.8093 |

---
## 🧩 Key Findings
- MEAN pooling consistently strongest and most stable.
- CLS pooling close on STS-B but less robust on STS12.
- MEAN+MAX competitive but failed to outperform MEAN → suggests redundancy/noise.
- MAX pooling weakest overall.
- MRPC consistently poor across all poolings → frozen embeddings struggle with fine-grained paraphrase alignment.
- Extended training monitoring shows:more epochs → lower loss but plateau/fluctuating validation accuracy → confirms risk of overfitting.


## Acknowledgements
This project replicates the work of:
Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks by Nils Reimers and Iryna Gurevych

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SentenceBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", pooling="mean"):
        super(SentenceBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling

        # if mean_max，output dimension double
        hidden_size = self.bert.config.hidden_size
        if pooling == "mean_max":
            hidden_size *= 2

        self.fc = nn.Linear(hidden_size, 1)

    def mean_pooling(self, token_embeddings, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

    def max_pooling(self, token_embeddings, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings = token_embeddings.clone()
        token_embeddings[mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, dim=1)[0]

    def cls_pooling(self, model_output):
        return model_output[0][:, 0, :]  # CLS token

    def forward(self, input_ids, attention_mask):
        model_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == "mean":
            pooled = self.mean_pooling(model_output[0], attention_mask)
        elif self.pooling == "max":
            pooled = self.max_pooling(model_output[0], attention_mask)
        elif self.pooling == "cls":
            pooled = self.cls_pooling(model_output)
        elif self.pooling == "mean_max":
            mean_emb = self.mean_pooling(model_output[0], attention_mask)
            max_emb = self.max_pooling(model_output[0], attention_mask)
            pooled = torch.cat([mean_emb, max_emb], dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return pooled

    def encode(self, sentences, batch_size=32, device='cpu'):
        self.eval()
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True,
                                           return_tensors='pt', max_length=128)
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            with torch.no_grad():
                embeddings = self.forward(encoded_input['input_ids'],
                                          encoded_input['attention_mask'])
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
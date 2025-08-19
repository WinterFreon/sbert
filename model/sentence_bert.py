import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SentenceBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, return_embeddings=True):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
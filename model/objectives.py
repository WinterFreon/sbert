import torch.nn as nn
import torch

class ClassificationObjective(nn.Module):
    def __init__(self, embedding_dim=768, num_labels=3):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim * 3, num_labels)

    def forward(self, u, v, labels):
        features = torch.cat([u, v, torch.abs(u - v)], dim=1)
        logits = self.classifier(features)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

class RegressionObjective(nn.Module):
    def forward(self, u, v, target_scores):
        cos_sim = torch.cosine_similarity(u, v)  # range [-1, 1]
        loss = nn.MSELoss()(cos_sim, target_scores)  # target_scores: [-1, 1]
        return loss

class TripletObjective(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        dist_pos = torch.norm(a - p, p=2, dim=1)
        dist_neg = torch.norm(a - n, p=2, dim=1)
        loss = torch.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()
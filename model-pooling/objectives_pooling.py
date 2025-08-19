import torch.nn as nn
import torch
import torch.nn.functional as F

class ClassificationObjective(nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super(ClassificationObjective, self).__init__()
        # embedding_dim is already the dimension after pooling
        self.classifier = nn.Linear(embedding_dim * 3, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, u, v, labels):
        features = torch.cat([u, v, torch.abs(u - v)], dim=1)
        logits = self.classifier(features)
        loss = self.loss_fct(logits, labels)
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
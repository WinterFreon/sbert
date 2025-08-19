config = {
    'model_name': 'bert-base-uncased',
    'embedding_dim': 768,
    'num_labels': 3,
    'batch_size': 16,
    'num_epochs': 1,
    'lr': 2e-5,
    'device': 'cuda',  # or 'cpu'
    'pooling': 'mean'  # pooling，[mean, max, cls, mean_max]
}

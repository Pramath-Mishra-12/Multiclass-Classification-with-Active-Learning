import torch
import numpy as np


# Minimum Margin-based sampling function
def minimum_margin_sampling(model_, unlabeled_loader, n_samples=10):
    model_.eval()
    margins = []

    with torch.no_grad():
        for batch in unlabeled_loader:
            inputs = {key: val.to('cuda' if torch.cuda.is_available() else 'cpu') for key, val in batch.items() if key != 'labels'}
            outputs = model_(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

            # Sort the probabilities for each sample and get the top two
            top_two_probs = torch.topk(probs, 2, dim=1).values
            # Compute the margin: difference between the highest and second-highest probabilities
            margin = top_two_probs[:, 0] - top_two_probs[:, 1]
            margins.append(margin.cpu().numpy())

    margins = np.array(margins)
    # Select n_samples with the smallest margin (highest uncertainty)
    uncertain_indices = margins.argsort()[:n_samples]
    return uncertain_indices

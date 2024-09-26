import torch
import numpy as np


# Entropy-based sampling function
def entropy_sampling(model_, unlabeled_loader, n_samples=10):
    model_.eval()
    entropies = list()

    with torch.no_grad():
        for batch in unlabeled_loader:
            inputs = {
                key: val.to('cuda' if torch.cuda.is_available() else 'cpu')
                for key, val in batch.items()
                if key != 'labels'
            }
            outputs = model_(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            # Calculate entropy: sum(p * log(p)) across all classes
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
            entropies.append(entropy.cpu().numpy())

    entropies = np.concatenate(entropies)
    # Select top n_samples with the highest entropy
    uncertain_indices = entropies.argsort()[-n_samples:]
    return uncertain_indices

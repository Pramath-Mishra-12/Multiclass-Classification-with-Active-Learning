import torch
import numpy as np


# Least Confidence Sampling function
def least_confidence_sampling(model_, unlabeled_loader_, n_samples=10):
    model_.eval()
    confidences = []

    with torch.no_grad():
        for batch in unlabeled_loader_:
            inputs = {
                key: val.to('cuda' if torch.cuda.is_available() else 'cpu')
                for key, val in batch.items()
                if key != 'labels'
            }
            outputs = model_(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

            # Calculate the highest class probability for each sample
            max_probs = torch.max(probs, dim=1).values
            confidences.extend(max_probs.cpu().numpy())

    confidences = np.array(confidences)
    # Select n_samples with the lowest confidence (most uncertain predictions)
    uncertain_indices = confidences.argsort()[:n_samples]
    return uncertain_indices

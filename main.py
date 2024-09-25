import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from sklearn.model_selection import train_test_split
from entropy_reduction_sampling import entropy_sampling
from datasets import load_dataset, concatenate_datasets
from minimum_margin_sampling import minimum_margin_sampling
from least_confidence_sampling import least_confidence_sampling
from transformers import BertTokenizer, BertForSequenceClassification


# Training function
def train_model(model_, train_loader):
    model_.train()
    optimizer = torch.optim.AdamW(model_.parameters(), lr=2e-5)

    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {key: val.to('cuda' if torch.cuda.is_available() else 'cpu') for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model_(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()


# Evaluation function (F1-Score)
def evaluate_model(model_, test_loader_):
    model_.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader_:
            inputs = {key: val.to('cuda' if torch.cuda.is_available() else 'cpu') for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model_(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Macro F1-Score: {f1:.4f}')
    return f1


# Active learning loop
def active_learning_loop(
        model_,
        labeled_dataset_,
        unlabeled_dataset_,
        test_dataset_,
        oracle_labels_,
        sampling_function,
        n_cycles=5,
        n_samples=10,
        batch_size=32
):
    for cycle in range(n_cycles):
        print(f"\nCycle {cycle + 1}/{n_cycles}")

        # Train the model
        train_loader = DataLoader(labeled_dataset_, batch_size=batch_size, shuffle=True)
        train_model(model_, train_loader)

        # Evaluate the model on the test set
        test_loader_ = DataLoader(test_dataset_, batch_size=batch_size, shuffle=False)
        evaluate_model(model_, test_loader_)

        # Perform least confidence sampling
        unlabeled_loader_ = DataLoader(unlabeled_dataset_, batch_size=batch_size, shuffle=False)
        uncertain_indices = sampling_function(model_, unlabeled_loader_, n_samples=n_samples)

        # Simulate the oracle labeling
        new_texts = [unlabeled_dataset_[i]['input_ids'] for i in uncertain_indices]
        new_labels = [oracle_labels_[i] for i in uncertain_indices]  # Simulated oracle labels

        # Add newly labeled data to the labeled dataset
        labeled_dataset_.texts.extend(new_texts)
        labeled_dataset_.labels.extend(new_labels)

        # Remove selected samples from the unlabeled dataset
        unlabeled_dataset_.texts = [
            text
            for i, text in enumerate(unlabeled_dataset_.texts)
            if i not in uncertain_indices
        ]



if __name__ == '__main__':
    # Tokenizer & Model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path='bert-base-uncased',
        num_labels=4
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load AG News dataset
    dataset = load_dataset("ag_news")
    dataset = concatenate_datasets(dataset['train'], dataset['test'])

    # Train-test split with stratification
    train_data, test_data = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset['label'],
        random_state=42
    )

    # Simulate oracle by further splitting train_data into labeled and unlabeled sets
    train_size = int(0.05 * len(train_data))
    labeled_data = train_data.select(range(train_size))
    unlabeled_data = train_data.select(range(train_size, len(train_data)))

    # Prepare data for labeled, unlabeled, and test sets
    labeled_texts = labeled_data['text']
    labeled_labels = labeled_data['label']
    unlabeled_texts = unlabeled_data['text']
    test_texts = test_data['text']
    test_labels = test_data['label']

    # Create datasets
    labeled_dataset = CustomDataset(labeled_texts, labeled_labels, tokenizer)
    unlabeled_dataset = CustomDataset(unlabeled_texts, None, tokenizer)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer)

    # Simulated oracle labels for unlabeled dataset
    oracle_labels = unlabeled_data['label']

    # Run the active learning loop
    active_learning_loop(
        model,
        labeled_dataset,
        unlabeled_dataset,
        test_dataset,
        oracle_labels,
        n_cycles=5,
        n_samples=10,
        batch_size=16,
        sampling_function=least_confidence_sampling
    )

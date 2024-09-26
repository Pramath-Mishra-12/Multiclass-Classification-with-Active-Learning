import torch
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from utils.custom_dataset import CustomDataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset, concatenate_datasets
from sampling.entropy_reduction import entropy_sampling
from sampling.minimum_margin import minimum_margin_sampling
from sampling.least_confidence import least_confidence_sampling
from transformers import BertTokenizer, BertForSequenceClassification


# Training function
def train_model(model_, train_loader):
    model_.train()
    optimizer = torch.optim.AdamW(model_.parameters(), lr=2e-5)

    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {key: val.to('cuda' if torch.cuda.is_available() else 'cpu') for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model_(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss / len(train_loader):.4f}")


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
        new_texts = [tokenizer.decode(unlabeled_dataset_[i]['input_ids']) for i in uncertain_indices]
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
        num_labels=14
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load Amazon product dataset
    dataset = load_dataset("iarbel/amazon-product-data-filter")
    print("Amazon product data loaded...")

    # Train dataset
    train_data = pd.DataFrame(concatenate_datasets([dataset['train'], dataset['validation']]))
    train_data = train_data[['title', 'category']]

    # Test dataset
    test_data = pd.DataFrame(dataset['test'])
    test_data = test_data[['title', 'category']]

    # Label Encoding
    le = preprocessing.LabelEncoder()
    le.fit(train_data['category'])
    train_data['category'] = le.transform(train_data['category'])
    test_data['category'] = le.transform(test_data['category'])
    print("Label encoding done...")

    # Simulate oracle by further splitting train_data into labeled and unlabeled sets
    labeled_data, unlabeled_data = train_test_split(
        train_data,
        test_size=0.8,
        stratify=train_data['category'],
        random_state=42
    )

    # Prepare data for labeled, unlabeled, and test sets
    labeled_texts = labeled_data['title'].tolist()
    labeled_labels = labeled_data['category'].tolist()
    unlabeled_texts = unlabeled_data['title'].tolist()
    test_texts = test_data['title'].tolist()
    test_labels = test_data['category'].tolist()

    # Create datasets
    labeled_dataset = CustomDataset(labeled_texts, labeled_labels, tokenizer)
    unlabeled_dataset = CustomDataset(unlabeled_texts, None, tokenizer)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer)

    # Simulated oracle labels for unlabeled dataset
    oracle_labels = unlabeled_data['category'].tolist()

    # Run the active learning loop
    active_learning_loop(
        model,
        labeled_dataset,
        unlabeled_dataset,
        test_dataset,
        oracle_labels,
        n_cycles=25,
        n_samples=10,
        batch_size=32,
        sampling_function=minimum_margin_sampling
    )

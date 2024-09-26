# Active Learning with Amazon Product Data

With the growth of data-centric Machine Learning, Active Learning has become increasingly popular among businesses and researchers. Active Learning seeks to progressively train machine learning (ML) models, reducing the amount of training data required to achieve competitive performance.

### Structure of an Active Learning Pipeline

An Active Learning pipeline typically consists of:
- **Classifier**: The ML model to be trained.
- **Oracle**: The oracle is responsible for labeling new data as needed by the model. The oracle can be a trained individual or a group of annotators who ensure consistency in the labeling process.

The process starts with:
1. **Initial Model Training**: A small subset of the full dataset is annotated and used to train an initial model. 
2. **Testing**: The model is then tested on a balanced test set. The performance of this initial model becomes the baseline.
3. **Iterative Training**: Based on business requirements, additional samples are labeled by the oracle, and the model is retrained with this expanded dataset.
4. **Stopping Criteria**: This cycle continues until an acceptable performance metric (e.g., accuracy, F1-Score) is reached or another business metric is met.

## Active Learning Sampling Techniques

This repository implements four different active learning sampling techniques, each demonstrated using the Amazon Product Data dataset:

1. **Least Confidence Sampling**: This strategy selects samples where the model has the least confidence in its predictions (i.e., the maximum predicted class probability is the lowest). Least confident samples are labeled and added to the training set.

2. **Entropy Reduction Sampling**: This approach selects samples that produce the highest entropy in the model's predictions. These are the samples about which the model is most uncertain and hence the most valuable to label.

3. **Minimum Margin Sampling**: In this method, the samples closest to the decision boundary are selected. The margin is defined as the difference between the top two predicted class probabilities. Smaller margins indicate higher uncertainty.

## Evaluation Metrics

The primary evaluation metric used in this project is the **F1-Score**. 

### What is the F1-Score?

The F1-Score is the harmonic mean of precision and recall, which makes it a useful measure when dealing with imbalanced datasets. It is defined as:

$$F1 = 2 * \frac {Precision * Recall} {Precision + Recall}$$

- **Precision**: The ratio of true positive predictions to the total positive predictions (true positives + false positives).
- **Recall**: The ratio of true positive predictions to the actual positives (true positives + false negatives).

### Why Use the F1-Score?

1. **Imbalanced Classes**: In multi-class classification tasks like Amazon Product Data, the distribution of classes may not be uniform. The F1-Score helps ensure that the model performs well on all classes, especially the minority ones.

2. **Balance Between Precision and Recall**: It provides a balance between precision and recall, giving a better sense of the model’s performance than accuracy alone, which can be misleading when classes are imbalanced.

3. **Model Comparison**: The F1-Score allows for effective comparison between different models and sampling strategies by providing a single metric that captures performance across multiple dimensions.

## Repository Structure

- `./utils/custom_dataset.py`: Implements the Custom Dataset Class.
- `./sampling/least_confidence_sampling.py`: Implements the Least Confidence Sampling technique.
- `./sampling/entropy_reduction_sampling.py`: Implements Entropy Reduction Sampling.
- `./sampling/minimum_margin_sampling.py`: Implements Minimum Margin Sampling.
- `main.py`: The main script that runs active learning loops for each sampling strategy using the Amazon Product Data dataset.

## Dataset

The **Amazon Product Data** dataset, from Hugging Face's `iarbel/amazon-product-data-filter`, is a multi-class text classification dataset. It contains product reviews and metadata from Amazon, categorized into **14 classes**. The dataset is loaded using the `datasets` library.

## Model

This repository uses a **BERT**-based model for text classification, leveraging the Hugging Face `transformers` library. BERT is pre-trained on large corpora and fine-tuned on the Amazon Product Data dataset for the classification task.

## Conclusion

This repository demonstrates various **Active Learning** sampling strategies on a multi-class classification task using the Amazon Product Data dataset. Each sampling strategy has its strengths, allowing you to experiment and compare which technique best suits your needs.
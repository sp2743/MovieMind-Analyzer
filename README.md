## 🌟 Introduction
This project, SentimentScope, is a hands-on implementation of sentiment analysis using a custom-built Transformer model in PyTorch. As a Machine Learning Engineer at CineScope, a growing entertainment company, the goal is to enhance the movie recommendation system by accurately classifying IMDB movie reviews as either positive (1) or negative (0).

Unlike typical text generation tasks where Transformers predict the next token, this project adapts the architecture for a binary classification task. Key differences include using subword tokenization (bert-base-uncased), applying a pooling mechanism (mean pooling) to condense all token embeddings into a single vector, and feeding that vector into a final classification head to generate class logits.

The model is trained using an epoch-based approach, making multiple full passes over the shuffled IMDB dataset to ensure robust learning and generalization.

## 📁 File Structure and Data
The core of the project is implemented within the SentimentScope.ipynb Jupyter Notebook.

Data Source
The dataset used is the IMDB Large Movie Review Dataset, provided in the aclIMDB_v1.tar.gz archive.

Directory Structure
The unzipped data follows a standard, organized structure:

```text
aclIMDB/
├── train/
│   ├── pos/    # Positive reviews (Label 1)
│   ├── neg/    # Negative reviews (Label 0)
│   ├── unsup/  # Unsupervised data (Not used)
├── test/
│   ├── pos/    # Positive reviews for testing (Label 1)
│   ├── neg/    # Negative reviews for testing (Label 0)
```

## 💻 Solution Overview
The solution involves loading and preprocessing the text data, defining a custom PyTorch Dataset and DataLoader, customizing the Transformer architecture, and training the model using a standard optimization loop.

### __1. Data Loading and Preparation:__

The raw text files from the train and test directories are loaded into Pandas DataFrames.

The training data (25,000 samples) is split into a Training Set (90%, 22,500 samples) and a Validation Set (10%, 2,500 samples) after shuffling to ensure balanced label distribution.

The Test Set (25,000 samples) is reserved for final evaluation.

### __2. Tokenization and DataLoader:__
The bert-base-uncased tokenizer is used for efficient subword tokenization.

A custom IMDBDataset class inherits from torch.utils.data.Dataset to handle text-to-token conversion and return the correct label.

DataLoader instances are created for the training, validation, and test sets to manage batching and shuffling (for training). A fixed maximum sequence length (MAX_LENGTH=128) is enforced via padding and truncation.

### __3. Transformer Architecture Customization:__

The DemoGPT class is a custom implementation of a Transformer encoder adapted for classification.

Embedding: Standard token and positional embeddings are used.

Transformer Blocks: Multiple layers (layers_num=6) of **Block**s are stacked, each containing a MultiHeadAttention (heads_num=8) layer and a FeedForward network.

Classification Head (Key Customization):

The sequence of token embeddings is aggregated into a single vector using mean pooling across the time dimension: x = x.mean(dim=1).

This pooled vector is passed to a final nn.Linear classifier that maps the embedding dimension (d_embed=128) to the number of classes (num_classes=2), yielding the classification logits.

### __4. Training and Evaluation:__

Loss Function: nn.CrossEntropyLoss is used for the binary classification task.

Optimizer: optim.AdamW is used for optimization.

Training Loop: The model is trained for 10 epochs.

Performance Metric: A dedicated calculate_accuracy function tracks Training and Validation Accuracy after each epoch.

### __🔮 Future Plan: Practical Deployment:__

The next phase of the SentimentScope project will focus on deploying the trained model into a practical, interactive tool for CineScope to analyze real-world sentiment immediately.

Implementation Platform: Develop an interactive web application using Streamlit.

Data Extraction: Implement functionality to extract (scrape) user reviews from the web (e.g., from movie pages or forums) when a user enters a movie title.

Real-time Analysis: Feed the extracted text reviews directly into the trained DemoGPT model to perform sentiment classification (Positive/Negative).

Sentiment Reporting: Generate a comprehensive report that:

Classifies the total number of positive and negative reviews found.

Provides a summary review or qualitative assessment of the overall sentiment.

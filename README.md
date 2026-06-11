# IMDB SimpleRNN Sentiment Classifier

This repository contains a CS478 deep learning notebook that builds and evaluates recurrent neural networks for binary sentiment classification using the IMDB movie review dataset. The project uses Keras to process tokenized review sequences, train a SimpleRNN model, evaluate predictions with validation accuracy and ROC AUC, and compare a baseline model with an improved model.

## Project Overview

The project demonstrates how recurrent neural networks can be used for text classification. The notebook loads tokenized IMDB movie reviews, pads each review to a fixed sequence length, uses an embedding layer to convert word IDs into trainable vectors, and applies a SimpleRNN layer to process the review sequence before predicting whether the review is positive or negative.

The notebook includes a baseline SimpleRNN model and an improved SimpleRNN model with adjusted hyperparameters, model checkpointing, early stopping, and learning rate reduction.

## Dataset

The project uses the Keras IMDB movie review dataset.

```text
Training reviews: 25,000
Validation reviews: 25,000
Task: Binary sentiment classification
Labels: 0 for negative review, 1 for positive review
Vocabulary size used: 10,000 words
Maximum review length: 100 tokens
Padding and truncation: Pre
```

## Preprocessing

The notebook prepares the IMDB review data by:

- Loading tokenized reviews from Keras
- Restricting the vocabulary to the 10,000 most common words
- Padding shorter reviews to 100 tokens
- Truncating longer reviews to 100 tokens
- Using binary sentiment labels for classification

## Baseline SimpleRNN Model

The baseline model uses an embedding layer, spatial dropout, a SimpleRNN layer, and a sigmoid output layer.

```text
Embedding dimension: 64
Vocabulary size: 10,000
Maximum review length: 100
Spatial dropout: 0.2
SimpleRNN units: 256
RNN dropout: 0.2
Output activation: Sigmoid
Loss function: Binary cross entropy
Optimizer: Adam
Epochs: 16
Batch size: 128
```

## Baseline Model Summary

```text
Embedding parameters: 640,000
SimpleRNN parameters: 82,176
Output parameters: 257
Total parameters: 722,433
Trainable parameters: 722,433
```

## Baseline Results

The baseline model reached its best validation accuracy at epoch 11.

```text
Baseline best validation accuracy: 79.01 percent
Baseline best epoch: 11
Base ROC AUC: 86.63
```

The baseline training curves show that training accuracy continued increasing while validation loss became unstable, which suggests overfitting.

## Improved SimpleRNN Model

The improved model keeps the same IMDB dataset and SimpleRNN approach, but changes the embedding size, RNN size, dropout, learning rate, and training callbacks.

```text
Embedding dimension: 128
Vocabulary size: 10,000
Maximum review length: 100
Spatial dropout: 0.3
SimpleRNN units: 128
RNN dropout: 0.3
Recurrent dropout: 0.2
Learning rate: 0.0005
Epochs: 20
Batch size: 64
Model checkpointing: Enabled
Early stopping: Enabled
ReduceLROnPlateau: Enabled
```

## Improved Model Summary

```text
Embedding parameters: 1,280,000
SimpleRNN parameters: 32,896
Output parameters: 129
Trainable parameters: 1,313,025
```

## Improved Model Results

The improved model performed better than the baseline model.

```text
Improved best validation accuracy: 81.02 percent
Improved best epoch: 10
Improvement over baseline: 2.00 percentage points
Improved ROC AUC: 88.32
```

## Key Takeaways

- SimpleRNN models can process tokenized movie reviews as sequences.
- Embedding layers create trainable word representations for sentiment classification.
- The baseline RNN showed signs of overfitting as training accuracy rose while validation metrics became unstable.
- The improved model increased validation accuracy from 79.01 percent to 81.02 percent.
- ROC AUC improved from 86.63 to 88.32.
- Model checkpointing, early stopping, and learning rate reduction helped preserve the best validation model.

## Visualizations

The notebook includes:

- Baseline training accuracy versus validation accuracy
- Baseline training loss versus validation loss
- Baseline prediction probability distribution
- Baseline ROC curve
- Improved training accuracy versus validation accuracy
- Improved training loss versus validation loss
- Improved prediction probability distribution
- Improved ROC curve

## Technologies Used

- Python
- Keras
- TensorFlow backend
- scikit-learn
- Matplotlib
- Jupyter Notebook
- Google Colab

## Repository Structure

```text
.
├── Jason_Stys_CS478_01_CA13_NLP_RNN.ipynb
├── Jason Stys CS478-01 CA13 NLP RNN-1.pdf
├── README.md
└── requirements.txt
```

Optional generated folders after running the notebook:

```text
model_output/
├── rnn_base/
└── rnn_improved/
```

## How to Run

Clone the repository.

```bash
git clone https://github.com/your-username/imdb-simple-rnn-sentiment-classifier.git
cd imdb-simple-rnn-sentiment-classifier
```

Install the required packages.

```bash
pip install tensorflow keras matplotlib scikit-learn notebook
```

Launch Jupyter Notebook.

```bash
jupyter notebook
```

Open the notebook file and run all cells from top to bottom.

## Skills Demonstrated

- Natural language processing with tokenized review sequences
- IMDB sentiment classification
- Keras dataset loading
- Sequence padding and truncation
- Embedding layer design
- Spatial dropout
- SimpleRNN sequence modeling
- Binary classification with sigmoid output
- Model checkpointing
- Early stopping
- Learning rate scheduling
- ROC AUC evaluation
- ROC curve visualization
- Training and validation curve analysis
- Jupyter Notebook workflow

## Possible Future Improvements

- Compare SimpleRNN performance against LSTM and GRU models
- Add confusion matrix and precision recall evaluation
- Increase the maximum review length
- Tune vocabulary size and embedding dimension
- Add bidirectional recurrent layers
- Use pretrained word embeddings
- Save the best trained model for reuse

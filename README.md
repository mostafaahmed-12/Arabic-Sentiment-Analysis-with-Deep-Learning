# Arabic Sentiment Analysis with Deep Learning

## Overview

This project focuses on sentiment analysis for Arabic text using deep learning techniques, specifically employing transformers, recurrent neural networks (RNN), and convolutional neural networks (CNN). The goal is to build a robust model that accurately classifies Arabic text into different sentiment classes, such as positive, negative, or neutral.

## Table of Contents

- [Key Features](#key-features)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Experimental Approaches](#experimental-approaches)
- [Results](#results)


## Key Features

- Preprocessing techniques for Arabic text.
- Implementation of LSTM, Transformer, and CNN models.
- Handling class imbalance and resampling techniques.
- Training and evaluation strategies.
- Visualizations and analysis tools.


## Data Preparation
The dataset is preprocessed as follows:

- Duplicated reviews are removed from the training data.
- Textual data is processed by replacing and removing specific characters, handling diacritics, and normalizing the text.
- Stop words are removed, and stemming is applied.
- Emojis are handled, and an embedding is created.
- One-hot encoding is performed for target classes.

Model Architectures
LSTM Model
Embedding layer with 10,000 words and an output dimension of 128.
LSTM layer with 128 units.
Dense layer with softmax activation for classification and three neurons.
Transformer Model
Custom transformer block with multi-head self-attention and feed-forward layers.
Global average pooling for feature extraction.
Dense layers for classification.
CNN Model
Embedding layer with 10,000 words and an output dimension of 100.
LSTM layer with 50 units and dropout of 0.2.
Conv1D layer with 100 filters, kernel size of 3, and ReLU activation.
GlobalMaxPooling1D layer.
Dense layers with ReLU activation and dropout

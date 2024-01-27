# Project Title

Deep Learning-Based Sentiment Analysis for Arabic Text

## Overview

This project focuses on sentiment analysis for Arabic text using deep learning techniques, including transformers, recurrent neural networks (RNN), and convolutional neural networks (CNN). The goal is to build a robust model that accurately classifies Arabic text into different sentiment classes, such as positive, negative, or neutral.

## Table of Contents

- [Key Features](#key-features)
- [Data Preparation](#data-preparation)
- [Text Tokenization and Encoding](#Text-Tokenization-and-Encoding)
- [Imbalance Handling](#Imbalance-Handling)
- [Model Architectures](#model-architectures)
- [Training and Evaluation](#training)
- [Experimental Approaches](#experimental-approaches)
- [Results](#results)


## Key Features

- Preprocessing techniques for Arabic text.
- Implementation of LSTM, Transformer, and CNN models.
- Handling class imbalance and resampling techniques.
- Training and evaluation strategies.
- Visualizations and analysis tools.

## Data Preparation
 - Duplicated reviews are removed from the training data.
 - Textual data is processed by replacing and removing specific characters, handling diacritics, and normalizing the text.
 - Stop words are removed, and stemming is applied.
 - Emojis are handled, and an embedding is created.
 - One-hot encoding is performed for target classes.

## Text Tokenization and Encoding
 - The Arabic text is tokenized using the Tokenizer with a vocabulary size of 10,000 words.
 - Encoding schemes are applied to represent the text in a format suitable for deep learning models.

## Imbalance Handling
  - Techniques like resampling are employed to address class imbalance in the dataset.
  -oversampling ,undersampling and data augmentation

## Model Architectures
   - Two main deep learning architectures are explored: LSTM-based, Transformer-based, and a custom CNN model.
   - LSTM-based Model:
      - Embedding layer with 10,000 words and an output dimension of 128.
      - LSTM layer with 128 units.
      - Dense layer with softmax activation for classification and three neurons.
   - Transformer-based Model:
      - Custom transformer block with multi-head self-attention and feed-forward layers.
      - Global average pooling for feature extraction.
      - Dense layers for classification.
   - CNN Model:
      - Embedding layer with 10,000 words and an output dimension of 100.
      - LSTM layer with 50 units and dropout of 0.2.
      - Conv1D layer with 100 filters, kernel size of 3, and ReLU activation.
      - GlobalMaxPooling1D layer.
      - Dense layers with ReLU activation and dropout.
 ## Training and Evaluation
   - The models are trained using categorical cross-entropy loss and the Adam optimizer.
   - Early stopping is employed to prevent overfitting.
   - Validation is performed on a 20% split of the data in each epoch.
   - Learning rate schedules and custom callbacks are used for optimization.
 ## Experimental Approaches
 - Different experiments are conducted, including handling class imbalance, resampling data, adjusting hyperparameters, and incorporating regularization techniques.
 - Model performance metrics such as accuracy and categorical cross-entropy loss are monitored and evaluated.
 - optimization  Algorithm is Adam
 ## Experimental Approaches
 -LSTM 80%.
 - Transfromers 81%.
 - CNN2LSTN 82%.











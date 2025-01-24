# Darija-Sentiment-Analysis

A Python module for performing sentiment analysis on tweets written in Moroccan Darija. The module is built using TensorFlow and includes functionalities for preprocessing, training, evaluating, and testing a sentiment analysis model.

## Features

- Preprocess tweets by tokenizing and padding text data.
- Build a bidirectional LSTM model with dropout regularization for binary classification (positive or negative sentiment).
- Train the model with learning rate scheduling and evaluate its performance.
- Visualize training performance and confusion matrices.
- Test the model on custom sentences to predict sentiment and confidence scores.

## Dataset

The dataset used is a collection of tweets in Moroccan Darija across various domains:
- Initially formatted as `MSAC.arff` and converted to a `CSV` file for easier use.
- Labels are encoded as `1` (Positive) and `0` (Negative).
- Cleaned and balanced dataset with a total of **2000 tweets**:
  - **1000 positive**
  - **1000 negative**

## How to Use
the model tkinzer and the saved model are provided you only need to install them and call them in a specific task !

### Prerequisites

Make sure you have Python 3.7+ installed along with the following libraries:

- `tensorflow`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

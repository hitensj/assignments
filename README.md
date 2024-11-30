**Stock Market Analysis and Prediction**

This project leverages machine learning models to analyze stock market data, predict stock prices, and classify trading volumes. It demonstrates using K-Nearest Neighbors (KNN), Linear Regression, and Logistic Regression models, supported by comprehensive visualizations.

**Table of Contents**

Overview

Dataset

Features

Machine Learning Models

Results

Visualizations

How to Run

Future Enhancements

**Overview**

This project aims to:

Predict the "Close" price of stock indices using regression models.
Classify trading volume into "High" or "Low" categories using classification techniques.
Visualize the performance of the models to understand the results better.

**Dataset**

The dataset contains historical stock index data with the following columns:

Index: Name of the stock market index.

Date: Trading date.

Open, High, Low, Close, Adj Close: Stock prices.

Volume: Trading volume.

Preprocessing:

Missing values were imputed.

The categorical column Index was encoded using Label Encoding.
Features were scaled using StandardScaler.

**Features**

The following features were used to build the models:

Index (Encoded as numeric values)
Open
High
Low
Adj Close
Volume
**Machine Learning Models**

*1. K-Nearest Neighbors (KNN)*

Used for predicting "Close" prices.
Hyperparameter tuning of n_neighbors was conducted using Grid Search.

*2. Linear Regression*

Used for predicting "Close" prices.
No hyperparameter tuning was necessary, as it's a straightforward regression model.

*3. Logistic Regression*

They are used for classifying "High" or "Low" trading volumes.
Hyperparameter tuning of regularization strength C and penalty type was performed using Grid Search.

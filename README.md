# Performance Analysis of Stocks Using Deep Learning Models

This project focuses on predicting stock price changes using various deep learning models. Specifically, it compares the performance of Multilayer Perceptron (MLP), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) models in predicting the stock prices of Apple Inc. (AAPL). The aim is to identify the most effective model for accurate stock price forecasting, which can be valuable for investors and financial analysts.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Project Overview

Stock price prediction is a critical task in the financial domain, with implications for investors and traders. This project leverages deep learning techniques to forecast stock prices, specifically focusing on Apple's stock data. By comparing three different neural network architectures—MLP, LSTM, and GRU—the project aims to determine which model is best suited for this task.

## Dataset

- **Source**: Yahoo Finance
- **Stock**: Apple Inc. (AAPL)
- **Date Range**: July 15, 2005 - May 15, 2023
- **Columns**:
• Open - Open price of a stock is the initial price at which it starts trading at the beginning of a trading day.
• High - The high price refers to the maximum price at which the stock was traded throughout a trading day.
• Low - The low price is the minimum price at which the stock was traded throughout a trading day.
• Close - The Close Price is the last price at which a stock is traded on a particular day.
• Volume - Volume is measured as the total number of shares bought and sold during a given trading period.

The dataset consists of 4489 rows and 7 columns, capturing the daily stock prices of Apple Inc.

## Methodology

### Data Preprocessing

1. **Handling Missing Values**: Checked and confirmed no missing values in the dataset.
2. **Date Formatting**: Converted the 'Date' column to DateTime format and set it as the index for time-series analysis.
3. **Feature Selection**: Used the 'High' column to train the models, as it reflects the stock's best performance.
4. **Data Splitting**: The dataset was split into an 85% training set and a 15% test set.
5. **Scaling**: Applied MinMaxScaler for feature scaling to normalize the data.

### Model Implementation

- **Multilayer Perceptron (MLP)**: A basic feedforward neural network with multiple hidden layers.
- **Long Short-Term Memory (LSTM)**: A recurrent neural network (RNN) architecture designed to capture long-term dependencies in sequential data.
- **Gated Recurrent Unit (GRU)**: A simpler alternative to LSTM with fewer parameters and faster training times.

### Evaluation Metrics

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

## Models Implemented

### 1. Multilayer Perceptron (MLP)
- Architecture: Multiple hidden layers with non-linear activation functions.
- Purpose: To capture complex patterns in stock price data.

### 2. Long Short-Term Memory (LSTM)
- Architecture: Consists of memory cells, input, forget, and output gates.
- Purpose: To model long-term dependencies in time-series data.

### 3. Gated Recurrent Unit (GRU)
- Architecture: Contains reset and update gates for simpler memory management.
- Purpose: To achieve similar performance to LSTM with a more straightforward structure.

## Results

### MLP
- Initial accuracy: 40% with 10 epochs.
- Improved accuracy: 73% with 50 epochs.

### LSTM
- Initial accuracy: 71% with 10 epochs.
- Improved accuracy: 83% with 50 epochs.

### GRU
- Initial accuracy: 69% with 10 epochs.
- Improved accuracy: 79% with 50 epochs.

## Conclusion

The LSTM model demonstrated superior performance in predicting stock prices, capturing complex and long-term patterns more effectively than MLP and GRU models. This makes LSTM the preferred model for stock price prediction tasks, particularly in scenarios requiring the handling of sequential data with long-term dependencies.

## Research Paper

- [Performance Analysis of Stocks using Deep Learning Models](https://authors.elsevier.com/sd/article/S1877-0509(24)00624-0)

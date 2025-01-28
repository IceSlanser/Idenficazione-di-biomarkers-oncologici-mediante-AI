# Identificazione di Biomarcatori Oncologici mediante AI

## Overview
**Identificazione di Biomarcatori Oncologici mediante AI** aims to apply Artificial Intelligence and Machine Learning techniques to identify potential biomarkers in favor of oncological diseases diagnose. By combining data preprocessing, and classification modeling via XGBoost this project provides an end-to-end workflow for biomarker discovery.

### Disclaimer
This project is for research purposes and does not constitute medical advice or diagnostics. Any results must be validated by qualified medical personnel and supported by further experimental analyses.

## Features
- **Data Preprocessing**: Normalization and standardization, optimization, outlier detection and removal, handling of missing values.
- **Modeling**: Classification using XGBoost, hyperparameter tuning with Optuna using Tree-structured Parzen Estimator(TPE).
- **Interpretability**: Generation of a decision tree to illustrate classification logic.

## Prerequisites
- **Python 3.7 or higher**
- **Required libraries**: Numpy, Pandas, Scikit-learn, XGboost, Matplotlib, Seaborn, Optuna, Graphviz

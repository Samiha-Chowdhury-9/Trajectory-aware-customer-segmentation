# Trajectory-Aware Customer Segmentation

Customer segmentation project using **dynamic RFM features, GRU-based Sequential VAE, GMM clustering, churn prediction, and SHAP interpretability**.

## Tech Stack

<p>   <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40" title="Python"/>   <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" width="40" title="Jupyter"/>   <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="40" title="Pandas"/>   <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="40" title="NumPy"/>   <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="40" title="PyTorch"/>   <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="40" title="Scikit-learn"/>   <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matplotlib/matplotlib-original.svg" width="40" title="Matplotlib"/> </p>

**Also used:** LightGBM, SHAP, Seaborn

------

## Overview

Traditional RFM segmentation represents customers using a single static snapshot.

This project explores whether customer behavior over time can be modeled more effectively using monthly RFM sequences and a **Sequential Variational Autoencoder (VAE)**.

The learned customer embeddings were clustered and evaluated on a downstream churn prediction task.

------

## Workflow

```text
Transaction Data
      ↓
Data Cleaning
      ↓
Monthly RFM Features
      ↓
Customer Sequences
      ↓
GRU-based Sequential VAE
      ↓
Latent Embeddings
      ↓
GMM Clustering
      ↓
Churn Prediction
      ↓
SHAP Interpretation
```

------

## Dataset

Two e-commerce datasets were examined:

- **Olist Brazilian E-Commerce**
- **Online Retail II**

Olist contained too few customers with sufficiently long purchasing histories for trajectory modeling.

The final experiments therefore used **Online Retail II**.

After preprocessing:

- **805,620 transactions**
- **5,881 customers**
- **3,030 customers** active across at least 3 months

------

## Feature Engineering

Monthly customer sequences were created using:

- Recency
- Frequency
- Monetary Value
- Average Order Value
- Quantity
- Time-decayed Frequency

The final sequence shape was:

```text
3030 customers × 22 months × 6 features
```

------

## Sequential VAE

The main model used:

- Bidirectional GRU encoder
- 2 GRU layers
- 64 hidden units
- 8-dimensional latent space
- GRU decoder
- Weighted reconstruction loss
- KL-divergence regularization
- Activity masking

Implemented using **PyTorch**.

------

## Models Compared

Three customer representations were evaluated:

1. **Static RFM + K-Means**
2. **Static VAE + K-Means**
3. **Sequential VAE + GMM**

For the fixed `k = 5` clustering experiment:

| Method               | Silhouette ↑ | Davies-Bouldin ↓ |
| -------------------- | ------------ | ---------------- |
| Trajectory VAE + GMM | 0.1482       | 2.0371           |
| Static K-Means       | 0.5457       | **0.7586**       |
| Static VAE + K-Means | **0.5598**   | 0.7611           |

------

## Churn Prediction

Customer inactivity during the final holdout period was used as the downstream target.

### Logistic Regression

| Features       | ROC-AUC    |
| -------------- | ---------- |
| Static RFM     | 0.6840     |
| Static VAE     | **0.6903** |
| Sequential VAE | 0.6805     |

### LightGBM

| Features       | ROC-AUC    |
| -------------- | ---------- |
| Static RFM     | **0.7769** |
| Sequential VAE | 0.7145     |

The sequential model did not outperform the simpler static baselines in the current experiment.

------

## Explainability

**SHAP** was used to analyze how the learned latent dimensions influenced churn prediction.

A Random Forest surrogate model was also used to approximate the customer clusters and identify influential original features.

------

## Repository Structure

```text
notebooks/
├── 01_data_and_viability.ipynb
├── 02_feature_engineering.ipynb
├── 03_sequential_vae.ipynb
├── 04_clustering_and_baselines.ipynb
├── 05_external_validation.ipynb
└── 06_interpretability_and_results.ipynb
```

------

## Skills Demonstrated

- Data Cleaning & EDA
- RFM Feature Engineering
- Temporal Data Modeling
- PyTorch
- GRU Networks
- Variational Autoencoders
- K-Means & GMM Clustering
- Logistic Regression
- LightGBM
- Cross-Validation
- Model Evaluation
- SHAP Explainability
- Baseline Comparison

------

## Key Finding

The experiment showed that a more complex sequential model does not automatically outperform simpler representations.

In this dataset, **static RFM features and static VAE representations produced stronger clustering and churn-prediction results** than the Sequential VAE.

This project demonstrates an end-to-end machine learning workflow including **data preparation, deep learning, clustering, validation, explainability, and critical evaluation of model performance**.

------

## Author

**Samiha Chowdhury**

Computer Science | Data Science | Machine Learning | Artificial Intelligence
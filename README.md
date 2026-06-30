# Credit Card Fraud Detection

A machine learning model that detects fraudulent credit card transactions using XGBoost, trained on the Sparkov synthetic fraud dataset (~1.85M transactions, Jan 2019 - Dec 2020).

## Overview

This project predicts whether a credit card transaction is fraudulent based on transaction amount, time of day, merchant category, and merchant identity. The model addresses common challenges in fraud detection: severe class imbalance (fraud is ~0.5% of transactions), the need for model explainability, and avoiding overfitting on features that don't generalize across time.

## Key Results

| Metric | Score |
|---|---|
| Fraud Precision | 0.77 |
| Fraud Recall | 0.65 |
| Fraud F1-score | 0.70 |
| Normal Precision/Recall | 1.00 / 1.00 |
| Overall Accuracy | 1.00 |

*Note: precision/recall numbers above were obtained by tuning the classification threshold (0.959) for higher precision. The model can alternatively be tuned for higher recall (~0.93) at the cost of more false positives — see "Threshold Tradeoff" below.*

## How It Works

1. **Data cleaning** — loads transaction data, extracts the hour of each transaction from the timestamp, and drops identifying columns (card number, name, address) that wouldn't generalize to new data.
2. **Feature engineering** — uses target encoding on `merchant` and `category` (mapping each category to its historical fraud rate) instead of plain label encoding, so the model can handle categories it didn't see in training.
3. **Handling class imbalance** — instead of generating synthetic fraud examples (SMOTE), the model uses `scale_pos_weight` in XGBoost to make the model pay more attention to the rare fraud class during training.
4. **Final feature set** — `amt`, `hour`, `category`, `merchant`. These four features were selected after debugging an overfitting issue (see below).
5. **Threshold tuning** — instead of using the default 0.5 cutoff, the decision threshold is tuned using precision-recall curves to balance false positives against missed fraud.
6. **Explainability** — SHAP (SHapley Additive exPlanations) is used to show which features drove each individual prediction, rather than treating the model as a black box.

## The Overfitting Debugging Story

An early version of this model used many more features (`job`, `city`, `state`, `gender`, `dob`, etc.) and looked excellent on training data (85-100%) but collapsed to ~14% fraud recall on test data. Investigation traced this to **data drift**: target-encoding columns like `job`, `city`, `state`, and `gender` baked in patterns specific to the training period (Jan 2019 onward) that simply didn't hold for the test period (later in 2020). Stripping the model down to four genuinely predictive, time-stable features (`amt`, `hour`, `category`, `merchant`) fixed the generalization gap. This is the core lesson of the project: **more features is not automatically a better model**, especially with time-based train/test splits.

## Threshold Tradeoff

The fraud detection threshold controls a precision/recall tradeoff:

- **Higher threshold (e.g. 0.959)** → fewer false alarms, but misses more real fraud. Better suited for automatically blocking transactions, where falsely declining a real customer is costly to trust.
- **Lower threshold** → catches more fraud (recall up to ~93%) but with more false positives. Better suited for flagging transactions for human review, where a person double-checks anyway.

This is a business decision, not a purely technical one — the right threshold depends on whether false positives or false negatives are more costly in the deployment context.

## Tech Stack

- Python, pandas, numpy
- XGBoost (classifier)
- category_encoders (target encoding)
- scikit-learn (scaling, metrics, threshold tuning)
- SHAP (model explainability)
- matplotlib / seaborn (visualization)

## Limitations & Future Work

See [LIMITATIONS.md](LIMITATIONS.md) for a full discussion of the model's current limitations and ideas for improvement.

## Dataset

[Sparkov Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle) — synthetic but realistic transaction data covering Jan 2019 - Dec 2020.

## How to Run

```bash
pip install -r requirements.txt
```

Then run the notebook cells in order. Requires `fraudTrain.csv` and `fraudTest.csv` in the project directory (download from the Kaggle link above).

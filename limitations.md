# Limitations & Future Work

This project is a learning/portfolio implementation, not a production fraud detection system. Being upfront about its limitations:

## Modeling Limitations

- **Only 4 features.** The final model uses `amt`, `hour`, `category`, and `merchant`. Real-world fraud systems typically also use behavioral signals this model lacks, such as: distance between consecutive transaction locations, transaction velocity (number of transactions in a short window), whether this is a new merchant for this card, and device/IP fingerprinting.
- **Recall/precision tradeoff is unresolved by design.** At the tuned threshold, the model misses ~35% of real fraud (recall 0.65) in exchange for fewer false positives (precision 0.77). A different threshold trades these the other way. There's no single "best" answer without knowing the real business cost of each type of error.
- **Class imbalance is extreme** (fraud is ~0.4% of transactions), so small shifts in threshold or features can swing reported metrics significantly. Metrics should be interpreted with this in mind rather than treated as fully stable.
- **No adversarial robustness.** The model learned static historical fraud patterns. It has no mechanism to adapt if fraudsters change behavior specifically to evade detection.

## Data Limitations

- **Data drift.** Fraud patterns in the training period (early in the dataset) didn't fully match the test period (later in the dataset), as discovered during debugging. A static, never-retrained model is likely to degrade in accuracy over time as real-world fraud patterns evolve.
- **Synthetic dataset.** The Sparkov dataset is simulated, not real transaction data. Real consumer transaction data may have different statistical properties, noise patterns, and fraud strategies.

## Engineering Limitations

- **Not real-time.** This is a batch/offline model trained and evaluated on static CSV files. A production system would need a serving pipeline that scores transactions in milliseconds as they happen.
- **No retraining pipeline.** There's no automated process to retrain the model on new data as patterns shift over time.
- **No monitoring.** A deployed fraud model needs ongoing monitoring for performance decay, data drift, and fairness across customer segments — none of which is implemented here.

## Explainability Limitations

- **SHAP explains learned patterns only.** It can show which features drove a specific prediction, but it cannot surface entirely new fraud patterns the model never encountered during training.

## Future Improvements

- Add behavioral/velocity features (transactions per hour, distance from last transaction, etc.)
- Build a small Flask/FastAPI service to simulate real-time scoring
- Add periodic retraining logic and drift detection
- Experiment with ensemble methods or anomaly detection (e.g., Isolation Forest) as a complementary signal alongside the supervised model
- Evaluate fairness/bias across different customer demographics

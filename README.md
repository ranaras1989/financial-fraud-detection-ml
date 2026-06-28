# Financial Fraud Detection Using Machine Learning and XGBoost

## Project Overview

This project builds a machine learning model to identify fraudulent financial transactions using the PaySim synthetic mobile money transaction dataset.

Fraud detection is a highly imbalanced classification problem: most transactions are legitimate, while fraud cases are rare. Because of that, this project focuses on precision, recall, F1-score, confusion matrix results, and threshold tuning instead of relying only on accuracy.

The workflow compares a baseline Logistic Regression model with XGBoost, then improves the final XGBoost model using hyperparameter tuning and probability threshold optimization.

## Business Problem

In financial fraud detection, false negatives are costly because they represent fraudulent transactions that were missed by the model. At the same time, too many false positives can create unnecessary manual reviews.

The goal of this project is to:

- Detect as many fraud cases as possible
- Reduce false negatives
- Keep false positives reasonably low
- Improve recall, precision, and F1-score
- Choose a practical probability threshold for fraud prediction

## Dataset

Dataset used: PaySim Synthetic Financial Transaction Dataset

The dataset contains simulated mobile money transactions with features such as:

- Transaction type
- Transaction amount
- Sender old and new balance
- Receiver old and new balance
- Existing fraud flag
- Fraud label

The dataset file is not included in this repository because of file size. Download the PaySim CSV separately and place it in a `data/` folder:

```text
data/PS_20174392719_1491204439457_log.csv
```

Due to local memory constraints, the notebook uses the first 100,000 rows.

## Project Workflow

1. Load the dataset
2. Explore the data structure
3. Check fraud and non-fraud class distribution
4. Drop unnecessary ID columns
5. Encode categorical variables
6. Split data into training and testing sets
7. Train a baseline Logistic Regression model
8. Train an untuned XGBoost model
9. Handle class imbalance using `scale_pos_weight`
10. Evaluate models with precision, recall, F1-score, and confusion matrix
11. Perform cross-validation
12. Tune XGBoost hyperparameters with `RandomizedSearchCV`
13. Tune the probability threshold
14. Select the final model based on business tradeoffs

## Models Used

### Baseline Logistic Regression

Logistic Regression was used as a simple baseline model. It achieved high accuracy, but it failed to detect most fraud cases, showing why accuracy is not enough for imbalanced fraud detection.

### Untuned XGBoost

XGBoost performed much better than Logistic Regression and detected most fraud cases, but it produced more false positives.

### Tuned XGBoost

The final model used XGBoost with hyperparameter tuning and a custom probability threshold. This produced the best balance between recall and false positives.

## Final Model

Final selected model: **Tuned XGBoost with probability threshold = 0.4**

### Final Confusion Matrix

| Metric          |  Value |
| --------------- | -----: |
| True Negatives  | 19,968 |
| False Positives |      9 |
| False Negatives |      2 |
| True Positives  |     21 |

### Final Performance

| Metric    | Value |
| --------- | ----: |
| Precision | 70.0% |
| Recall    | 91.3% |
| F1-score  | 79.2% |

The final model correctly detected 21 out of 23 fraud cases while producing only 9 false positives.

## Model Comparison

| Model                         | False Positives | False Negatives | True Positives | Precision | Recall | F1-score |
| ----------------------------- | --------------: | --------------: | -------------: | --------: | -----: | -------: |
| Baseline Logistic Regression  |               0 |              22 |              1 |     1.000 |  0.043 |    0.083 |
| Untuned XGBoost               |              85 |               2 |             21 |     0.198 |  0.913 |    0.326 |
| Tuned XGBoost + Threshold 0.4 |               9 |               2 |             21 |     0.700 |  0.913 |    0.792 |

## Streamlit App

This repository includes a Streamlit app for real-time fraud risk prediction using the saved tuned XGBoost model.

To run the app locally:

```bash
streamlit run app.py
```

The app loads:

- `tuned_xgboost_fraud_model.pkl`
- `feature_columns.pkl`
- `final_threshold.pkl`

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## How to Run the Project

1. Clone this repository:

```bash
git clone https://github.com/ranaras1989/financial-fraud-detection-ml.git
cd financial-fraud-detection-ml
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the PaySim dataset from Kaggle.

4. Place the CSV file in the `data/` folder:

```text
data/PS_20174392719_1491204439457_log.csv
```

5. Open and run the notebook:

```text
FraudClassificationFinal.ipynb
```

6. Run the Streamlit app:

```bash
streamlit run app.py
```

## Repository Structure

```text
financial-fraud-detection-ml/
|-- FraudClassificationFinal.ipynb
|-- app.py
|-- tuned_xgboost_fraud_model.pkl
|-- feature_columns.pkl
|-- final_threshold.pkl
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## Key Learnings

- Accuracy can be misleading in imbalanced classification problems.
- Fraud detection should prioritize recall and false negatives.
- Logistic Regression is useful as a baseline model.
- XGBoost performs strongly on tabular fraud detection data.
- `scale_pos_weight` helps XGBoost handle class imbalance.
- Hyperparameter tuning can improve model performance.
- Threshold tuning helps align model behavior with business goals.

## Final Conclusion

The baseline Logistic Regression model achieved high accuracy but missed most fraud cases. XGBoost performed significantly better for this tabular fraud detection problem.

After hyperparameter tuning and threshold tuning, the final tuned XGBoost model achieved strong recall while keeping false positives low. The selected threshold of 0.4 detected 91.3% of fraud cases and provided a practical balance for a recall-sensitive fraud detection use case.

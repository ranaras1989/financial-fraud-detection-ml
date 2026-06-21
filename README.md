# Financial Fraud Detection using Machine Learning and XGBoost

## Project Overview

This project builds a financial fraud detection model using the PaySim synthetic financial transaction dataset.

The main goal is to detect fraudulent transactions in a highly imbalanced dataset. Since fraud cases are rare, this project focuses on evaluation metrics beyond accuracy, especially recall, precision, F1-score, confusion matrix, and threshold tuning.

The project compares a baseline Logistic Regression model with XGBoost and then improves the final model using hyperparameter tuning and probability threshold optimization.

---

## Business Problem

Financial fraud detection is a high-risk classification problem where missing fraud cases can lead to serious financial loss.

In this type of problem, accuracy alone can be misleading because the dataset is highly imbalanced. A model can achieve high accuracy by predicting most transactions as non-fraud, but still fail to detect actual fraud cases.

Therefore, this project focuses on:

* Detecting as many fraud cases as possible
* Reducing false negatives
* Keeping false positives low
* Improving precision, recall, and F1-score
* Selecting a business-friendly probability threshold

---

## Dataset

Dataset used: PaySim Synthetic Financial Transaction Dataset

The dataset contains simulated mobile money transactions and includes features such as:

* Transaction type
* Transaction amount
* Sender old and new balance
* Receiver old and new balance
* Fraud label

Due to local system memory constraints, this project uses the first 100,000 rows from the dataset.

The dataset file is not included in this repository because of file size. Please download the dataset separately and place the CSV file in the project folder.

---

## Project Workflow

The project follows a complete supervised machine learning workflow:

1. Load the dataset
2. Understand the data structure
3. Check class imbalance
4. Perform data preprocessing
5. Drop unnecessary ID columns
6. Encode categorical variables
7. Split data into training and testing sets
8. Train a baseline Logistic Regression model
9. Train an XGBoost model
10. Handle class imbalance using `scale_pos_weight`
11. Perform cross-validation
12. Tune hyperparameters using `RandomizedSearchCV`
13. Tune probability threshold
14. Evaluate final model using precision, recall, F1-score, and confusion matrix
15. Select the best final model

---

## Models Used

### 1. Baseline Logistic Regression

Logistic Regression was used as a simple baseline model.

The baseline model achieved very high accuracy, but it failed to detect most fraud cases. This showed that accuracy is not a reliable metric for highly imbalanced fraud detection problems.

### 2. Untuned XGBoost

XGBoost performed much better than Logistic Regression. It was able to detect most fraud cases while reducing false positives.

### 3. Tuned XGBoost

Hyperparameter tuning was performed using `RandomizedSearchCV` with `StratifiedKFold`.

The final tuned XGBoost model was selected as the best model.

---

## Final Model

Final selected model:

**Tuned XGBoost with probability threshold = 0.4**

### Final Confusion Matrix

| Metric          |  Value |
| --------------- | -----: |
| True Negatives  | 19,967 |
| False Positives |     10 |
| False Negatives |      2 |
| True Positives  |     21 |

### Final Performance

| Metric    | Value |
| --------- | ----: |
| Precision | 67.7% |
| Recall    | 91.3% |
| F1-score  | 77.8% |

The final model correctly detected 21 out of 23 fraud cases while producing only 10 false positives.

---

## Model Comparison

| Model                         | False Positives | False Negatives | True Positives | Precision | Recall | F1-score |
| ----------------------------- | --------------: | --------------: | -------------: | --------: | -----: | -------: |
| Baseline Logistic Regression  |               0 |              22 |              1 |     1.000 |  0.043 |    0.083 |
| Untuned XGBoost               |              85 |               2 |             21 |     0.198 |  0.913 |    0.326 |
| Tuned XGBoost + Threshold 0.4 |              10 |               2 |             21 |     0.677 |  0.913 |    0.778 |

---

## Key Learnings

This project demonstrates several important machine learning concepts:

* Accuracy can be misleading in imbalanced classification problems
* Fraud detection should focus strongly on recall and false negatives
* Logistic Regression is useful as a baseline model
* XGBoost performs strongly on tabular financial transaction data
* `scale_pos_weight` helps handle class imbalance in XGBoost
* Hyperparameter tuning can significantly improve model performance
* Threshold tuning is important for business-sensitive classification problems

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* Joblib

---

## How to Run the Project

1. Clone this repository

```bash
git clone <your-repository-url>
```

2. Install required dependencies

```bash
pip install -r requirements.txt
```

3. Download the PaySim dataset from Kaggle.

4. Place the CSV file in the project folder.

5. Open and run the notebook:

```text
fraud_detection_xgboost_project.ipynb
```

---

## Repository Structure

```text
financial-fraud-detection-ml/
│
├── fraud_detection_xgboost_project.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Final Conclusion

The baseline Logistic Regression model achieved high accuracy but failed to detect most fraud cases due to severe class imbalance.

XGBoost performed significantly better on this tabular fraud detection problem. After hyperparameter tuning and threshold tuning, the final tuned XGBoost model achieved a strong balance between fraud detection and false alarm reduction.

The final selected threshold of 0.4 was chosen because fraud detection is a recall-sensitive problem. This threshold allowed the model to detect 91.3% of fraud cases while keeping false positives very low.

This project shows a complete machine learning workflow from baseline modeling to final business-aware model selection.

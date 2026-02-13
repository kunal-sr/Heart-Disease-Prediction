# 🫀 Predicting Heart Disease – Kaggle Competition

This repository contains my solution for the **Heart Disease Prediction** competition on Kaggle.  
The objective of this competition is to predict whether a patient has heart disease based on clinical and health-related features.

---

## 📌 Problem Statement

Given patient health data, the goal is to build a machine learning model that predicts the probability of heart disease.

- Target column: `Heart Disease`
- Evaluation Metric: **ROC-AUC Score**
- Task Type: Binary Classification

---

## 📂 Project Structure

├── train.csv  
├── test.csv  
├── sample_submission.csv  
├── train.py  
├── submission.csv  
└── README.md  

---

## ⚙️ Approach

### 1️⃣ Reproducibility
- Set global random seed for consistency across runs.

### 2️⃣ Feature Engineering
Created interaction feature:
- `Age_Cholesterol = Age × Cholesterol`

This helps capture nonlinear relationships between age and cholesterol levels.

### 3️⃣ Cross Validation
- Used **Stratified K-Fold (5 folds)** to maintain class balance.
- Prevented data leakage and ensured robust validation.

---

## 🤖 Models Used

### 🔹 LightGBM
- `n_estimators = 5000`
- `learning_rate = 0.01`
- Early stopping: 200 rounds
- Evaluation metric: AUC

### 🔹 XGBoost
- `n_estimators = 5000`
- `learning_rate = 0.01`
- Early stopping: 200 rounds
- Evaluation metric: AUC

---

## 🔥 Model Blending

Final prediction is a weighted blend:

Final Prediction =  
0.6 × LightGBM + 0.4 × XGBoost  

Blending improved generalization and reduced variance.

---

## 📊 Validation Results

- LightGBM AUC: 
- XGBoost AUC:
- Final Blend AUC:

---

## 🛠️ How to Run

1. Install dependencies:
```
pip install numpy pandas scikit-learn lightgbm xgboost
```

3. Run training:
```
python train.py
```

4. Submission file will be generated:
```
submission.csv
```

---

## 💡 Key Learnings

- Importance of Stratified K-Fold in medical datasets  
- Power of gradient boosting models  
- Model blending improves leaderboard stability  
- Simple feature interactions can boost performance  

---

## 🚀 Future Improvements

- Hyperparameter optimization (Optuna)
- Feature importance analysis
- Advanced feature engineering
- Stacking instead of weighted blending

---

## 🏁 Kaggle Profile

You can check out the competition here:  
👉 https://www.kaggle.com/

---

## ⭐ If you found this useful
Give this repo a star ⭐ and feel free to connect!

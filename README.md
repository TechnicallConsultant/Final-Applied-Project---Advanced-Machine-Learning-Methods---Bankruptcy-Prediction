> **Trainee Name:** Fardus Qutaym  
> **Course Track:** Advanced Machine Learning Methods  
> **Academy:** SDAIA Academy  

---

# Cross-Country Corporate Bankruptcy Prediction Project

An applied machine learning project to predict corporate bankruptcy using tabular data, built as part of the SDAIA Academy course requirements.

## 📊 Dataset Source
The dataset used for this project is publicly available on Kaggle:
* **Dataset Name:** Cross-Country Corporate Bankruptcy Prediction Data
* **Kaggle Link:** [View Dataset on Kaggle](https://www.kaggle.com/datasets/shaikasif89/cross-country-corporate-bankruptcy-prediction-data)

---

## 🛠️ Project Phases & Workflow
1. **Environment Setup & Data Loading:** Initialized Google Colab, created local data directories, and ingested the 49,971 rows tabular dataset containing 9 financial features and a binary target (`Bankrupt`).
2. **Stratified K-Fold Validation:** Implemented 5-Fold Stratified Cross-Validation to maintain class distribution across splits.
3. **Baseline Model (LightGBM):** Built and trained a `LGBMClassifier` baseline model, achieving a strong **ROC-AUC of 0.8206**.
4. **Decision Threshold Tuning:** Performed an automated threshold sweep (from 0.01 to 0.99) to optimize performance under class imbalance, selecting an optimal threshold of **0.14** based on the F1-Score.
5. **Model Interpretability (SHAP):** Applied SHAP TreeExplainer to generate global and local summary plots, revealing that `Retained Earnings to Total Assets` and `Current Ratio` are the most influential predictive features.

---

## 🚀 How to Run the Code
1. Open the `project.ipynb` notebook in Google Colab.
2. Download the dataset from the Kaggle link provided above and place it in the `data/` directory.
3. Run the cells sequentially from data ingestion to SHAP analysis.

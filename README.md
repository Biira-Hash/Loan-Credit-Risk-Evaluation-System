# 💰 Loan Credit Risk Evaluation System

### 📊 Predicting Loan Default Risk Using Machine Learning

This project aims to automate and enhance the loan approval process by using machine learning to predict whether a loan applicant is likely to default or repay. The **Loan Credit Risk Evaluation System** provides a scalable, data-driven solution that improves accuracy, speed, and fairness in financial decision-making.

---

## 🧠 Project Overview

In the financial industry, accurately assessing credit risk is essential to minimize defaults and ensure sustainable lending. Traditional rule-based systems struggle with large datasets and complex relationships.  
This project leverages **Machine Learning (ML)** models — particularly **XGBoost, Random Forest, and Logistic Regression** — to predict credit risk based on applicant data.

The solution includes:
- Data preprocessing and feature engineering
- Model training and hyperparameter tuning
- Model evaluation and visualization
- Streamlit web app for real-time predictions

---

## 🚀 Features

✅ Automated credit risk prediction  
✅ Machine learning–based scoring (XGBoost, Random Forest, Logistic Regression)  
✅ Interactive Streamlit web app for loan approval simulation  
✅ Visual analytics for model insights  
✅ End-to-end pipeline from data ingestion to deployment  

---

## 🧩 System Architecture

The system processes raw loan data, trains models to classify applicants as **Low Risk (Approve)** or **High Risk (Reject)**, and deploys the best-performing model via a **Streamlit** interface.

---

## 🗂️ Dataset

**Source:** [Kaggle - Credit EDA Case Study](https://www.kaggle.com/datasets/venkatasubramanian/credit-eda-case-study?select=previous_application.csv)

The dataset includes features such as:
- Applicant demographics  
- Employment details  
- Credit history  
- Loan amount and purpose  
- Payment patterns  

---

## 🛠️ Technologies and Libraries

| Category | Tools/Libraries |
|-----------|----------------|
| **Data Processing** | `pandas`, `numpy`, `pyspark` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn`, `xgboost` |
| **Model Deployment** | `streamlit` |
| **Model Persistence** | `joblib` |

---

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows

3. Install Dependencies
pip install -r requirements.txt

4. Run Streamlit App
streamlit run app.py

🧮 Model Development
Algorithms Used

Logistic Regression: Simple, interpretable baseline model.

Random Forest: Strong performance with non-linear data.

XGBoost: Final model chosen for highest accuracy and AUC.

Model Metrics
Model	AUC	Accuracy	F1 Score
Logistic Regression	0.8811	0.8779	0.8556
Random Forest	0.9023	0.8874	0.8741
XGBoost	0.9046	0.8220	0.8812
Tuned XGBoost	0.9012	0.8187	0.8787
Model Output

xgboost_home_loan_model.pkl saved using joblib

Used for inference in the Streamlit web app

🧰 How to Use the Web App

Open the Streamlit app (app.py)

Enter applicant details (income, loan amount, credit history, etc.)

Click “Predict”

View result:

✅ Approved → Low credit risk

❌ Rejected → High credit risk

📈 Visualizations

Histograms & Boxplots – Loan distribution and variance

Heatmaps – Feature correlations

Countplots – Loan status by category

Feature Importance – Key drivers of default prediction

🧭 Future Scope

Integration with CIBIL/Experian APIs for credit score enrichment

Explainable AI with SHAP or LIME for model transparency

Real-time dashboards with Power BI / Tableau

Automated model retraining using Azure Data Factory / Databricks

Multi-class risk segmentation (Low, Medium, High)

👥 Team Members

Project Team 8 – PG-DBDA (Feb 2025)

Anandu Nair

Omkar Sawant

Ankit Surolia

Diptee Madekar

Rushikesh Chavan

Sadik Jamadar

Guide: Swapnil Adhav

📚 References

Understanding Logistic Regression – GeeksforGeeks

Understanding Random Forest – Analytics Vidhya

Math Behind XGBoost – Analytics Vidhya

Kaggle Dataset

🏁 Conclusion

The Loan Credit Risk Evaluation System demonstrates how ML can revolutionize loan approvals through automation and intelligent decision-making.
By combining accuracy, interpretability, and real-time usability, this project lays a foundation for modern, data-driven credit evaluation systems.

# 📊 Customer Churn Prediction Dashboard

An end-to-end Machine Learning project that predicts customer churn for a telecom company using multiple classification algorithms. The project includes an interactive Streamlit dashboard with Exploratory Data Analysis (EDA), customer churn prediction, explainable AI (SHAP), statistical analysis, and business recommendations.


## 🚀 Features

- Interactive EDA Dashboard
- Customer Churn Prediction
- Model Comparison (Logistic Regression, Random Forest, XGBoost)
- Model Performance Metrics
- SHAP Explainability
- Business Recommendations
- Statistical Analysis using Chi-Square Test
- Feature Engineering
- User-friendly Streamlit Interface


## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Explainable AI:** SHAP
- **Dashboard:** Streamlit
- **Model Persistence:** Joblib


## 📂 Project Structure
```
customer-churn-prediction/
│
├── app_streamlit.py              # Streamlit Dashboard
├── main.py                       # Main execution script
├── requirements.txt
├── README.md
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   └── evaluate.py
│
└── utils/
    └── recommendations.py
```

## 📌 Problem Statement

Customer churn is one of the biggest challenges faced by telecom companies. Losing existing customers directly impacts revenue and increases customer acquisition costs.

The objective of this project is to identify customers who are likely to churn so that the business can take proactive retention measures.

## 📊 Dataset

**Dataset:** Telco Customer Churn Dataset

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Customer Churn

Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn


## ⚙️ Feature Engineering

The following engineered features were created to improve model performance:

- Average Monthly Spend
- New Customer Flag
- High Bill Indicator
- Long-Term Customer Flag

These features help capture customer behavior beyond the original dataset.


## 🤖 Machine Learning Models

The project compares multiple machine learning algorithms:

- Logistic Regression
- Random Forest
- XGBoost

The best-performing model is automatically selected using cross-validation.


## 📈 Dashboard Modules

### 📊 Exploratory Data Analysis

- Churn Distribution
- Contract Type vs Churn
- Monthly Charges vs Churn
- Tenure Distribution


### 🔮 Prediction

Users can enter customer information to:

- Predict churn
- View churn probability
- Receive business recommendations
- Understand prediction using SHAP Explainability


### 📉 Model Performance

The dashboard displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve


### 📋 Statistical Analysis

Chi-Square tests are performed on important categorical variables to determine whether they significantly influence customer churn.

Business interpretations and recommendations are also provided for each analysis.


## 💡 Business Recommendations

Based on the predicted churn probability, the application suggests customer retention strategies such as:

- Offer long-term contract discounts
- Improve Fiber Optic service quality
- Provide loyalty rewards
- Offer personalized customer support
- Create bundled family plans

## 🔍 Explainable AI

The dashboard uses **SHAP (SHapley Additive Explanations)** to explain model predictions by identifying the features that contribute most to customer churn.


## ▶️ Installation

### Clone the repository

```bash
git clone https://github.com/Lahari468/customer-churn-prediction.git
```

### Navigate to the project

```bash
cd customer-churn-prediction
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```


## ▶️ Run the Project

### Train the models

```bash
python main.py
```

### Launch the Streamlit Dashboard

```bash
streamlit run app_streamlit.py
```


## 📈 Future Enhancements

- Deploy the application on Streamlit Community Cloud
- Add more feature engineering techniques
- Hyperparameter optimization using GridSearchCV
- Add customer segmentation using clustering
- Integrate real-time prediction API

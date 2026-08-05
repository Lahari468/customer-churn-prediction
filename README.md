# Customer Churn Prediction

A machine learning project to predict customer churn using Logistic Regression, Random Forest, and XGBoost. Includes a Streamlit app for predictions and EDA.

## Installation

1. Clone: `git clone https://github.com/Lahari468/customer-churn-prediction.git`
2. Create venv: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`

## Usage

- Train: `python main.py`
- Run app: `streamlit run app_streamlit.py`

## Dataset

Telco Customer Churn dataset from Kaggle.

## Project Structure

- `app_streamlit.py`: Streamlit app
- `main.py`: Training script
- `src/`: Preprocessing, training, evaluation
- `models/`: Saved models and artifacts
- `data/`: Dataset

## Technologies

Python, Scikit-learn, XGBoost, Streamlit, Pandas, Matplotlib, Seaborn, Joblib.

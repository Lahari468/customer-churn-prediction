import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from utils.recommendations import generate_recommendations
# ================== PAGE SETTINGS ==================
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

sns.set_style("whitegrid")

# ================== LOAD MODELS ==================
model = joblib.load("models/best_model.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")
scaler = joblib.load("models/scaler.pkl")
explainer = shap.Explainer(model)
raw_df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.markdown(
    """
    <h2 style='text-align:center;'>Customer Churn Prediction Dashboard</h2>
    <p style='text-align:center;'>End-to-end Machine Learning Application</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ================== TABS ==================
tab1, tab2, tab3,tab4 = st.tabs(
    ["📊 EDA Dashboard", "🔮 Prediction", "📉 Model Performance", "📈 Statistical Analysis"
]
)

# =====================================================================
# TAB 1 — EDA
# =====================================================================
with tab1:
    st.subheader("📊 Exploratory Data Analysis")

    df = raw_df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    st.write("Dataset Preview")
    st.dataframe(df.head(), width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### 🔹Churn Distribution")
        fig = plt.figure()
        sns.countplot(x="Churn", data=df)
        st.pyplot(fig)

    with col2:
        st.write("#### 🔹Churn vs Contract Type")
        fig = plt.figure()
        sns.countplot(x="Contract", hue="Churn", data=df)
        plt.xticks(rotation=10)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.write("#### 🔹Monthly Charges vs Churn")
        fig = plt.figure()
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
        st.pyplot(fig)

    with col4:
        st.write("#### 🔹Tenure Distribution by Churn")
        fig = plt.figure()
        sns.histplot(x="tenure", hue="Churn", data=df, kde=True, bins=30)
        st.pyplot(fig)


# =====================================================================
# TAB 2 — PREDICTION
# =====================================================================
with tab2:

    st.subheader("🔮 Customer Churn Prediction")

    st.info("Fill customer details below to predict customer churn.")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    with col2:
        tenure = st.slider("Tenure (Months)", 0, 72, 1)
        monthly = st.slider("Monthly Charges", 0.0, 200.0, 50.0)
        total = st.slider("Total Charges", 0.0, 10000.0, 500.0)

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    # -----------------------------
    # User Input DataFrame
    # -----------------------------

    user_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "InternetService": internet
    }])

    # ====================================================
    # Feature Engineering
    # ====================================================

    user_df["AverageMonthlySpend"] = (
        total / (tenure + 1)
    )

    user_df["NewCustomer"] = int(tenure < 6)

    user_df["HighBill"] = int(monthly > 80)

    user_df["LongTermCustomer"] = int(tenure >= 24)

    # -----------------------------
    # Encoding
    # -----------------------------

    encoded = pd.get_dummies(user_df, drop_first=True)

    for col in feature_cols:
        if col not in encoded.columns:
            encoded[col] = 0

    encoded = encoded[feature_cols]

    num_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AverageMonthlySpend"
    ]

    encoded[num_cols] = scaler.transform(
        encoded[num_cols]
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    if st.button("🚀 Predict"):

        prob = model.predict_proba(encoded)[0][1]
        pred = model.predict(encoded)[0]

        st.write("---")

        if pred == 1:

            st.error(
                f"""
                ❌ Customer is likely to churn

                **Probability:** {prob:.2%}
                """
            )

        else:

            st.success(
                f"""
                ✅ Customer is unlikely to churn

                **Probability:** {prob:.2%}
                """
            )

        # ==========================================
        # BUSINESS RECOMMENDATIONS
        # ==========================================

        st.write("---")
        st.subheader("💡 Recommended Business Actions")

        recommendations = generate_recommendations(
            user_df.iloc[0],
            prob
        )

        for rec in recommendations:
            st.success(rec)

        # ==========================================
        # SHAP EXPLANATION
        # ==========================================

        st.write("---")
        st.subheader("🔍 Why this prediction?")

        try:

            shap_values = explainer(encoded)

            fig = plt.figure(figsize=(10,6))

            shap.plots.waterfall(
                shap_values[0],
                max_display=10,
                show=False
            )

            st.pyplot(fig)

        except Exception as e:

            st.warning(
                "Unable to generate SHAP explanation."
            )

            st.code(str(e))

# =====================================================================
# TAB 3 — MODEL PERFORMANCE
# =====================================================================
with tab3:

    st.subheader("📉 Model Performance Metrics")

    df = raw_df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # ======================================================
    # Feature Engineering (MUST match training)
    # ======================================================
    df["AverageMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["NewCustomer"] = (df["tenure"] < 6).astype(int)
    df["HighBill"] = (df["MonthlyCharges"] > 80).astype(int)
    df["LongTermCustomer"] = (df["tenure"] >= 24).astype(int)

    # Keep only required columns
    df = df[
        [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "AverageMonthlySpend",
            "NewCustomer",
            "HighBill",
            "LongTermCustomer",
            "Contract",
            "InternetService",
            "Churn",
        ]
    ]

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Add missing columns expected by the model
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Arrange columns exactly as during training
    X = df[feature_cols]
    y = df["Churn"]

    # Scale numerical columns
    num_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AverageMonthlySpend",
    ]

    X[num_cols] = scaler.transform(X[num_cols])

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)


    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")
    col5.metric("ROC-AUC", f"{roc_auc:.2%}")


    st.write("### 🔹 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig = plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # -----------------------------
    # ROC Curve
    # -----------------------------
    st.write("### 🔹 ROC Curve")

    fpr, tpr, _ = roc_curve(y, y_prob)

    fig = plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)

# =====================================================================
# TAB 4 — STATISTICAL ANALYSIS
# =====================================================================
def chi_square_test(df, feature):

    contingency = pd.crosstab(df[feature], df["Churn"])

    chi2, p, dof, expected = chi2_contingency(contingency)

    st.write(f"## {feature} vs Churn")

    st.dataframe(contingency)

    col1, col2 = st.columns(2)

    col1.metric("Chi-Square", f"{chi2:.2f}")
    col2.metric("P-value", f"{p:.5f}")

    if p < 0.05:
        st.success(
            f"✅ {feature} has a statistically significant relationship with churn."
        )
    else:
        st.warning(
            f"❌ {feature} is not statistically significant."
        )

    # ===============================
    # Business Interpretation
    # ===============================

    if feature == "Contract":
        st.info("""
**Business Insight**

Customers on **month-to-month contracts** churn much more frequently than customers on yearly contracts.

**Recommendation**

Offer discounts or loyalty benefits for long-term subscriptions.
""")

    elif feature == "InternetService":
        st.info("""
**Business Insight**

Customers using **Fiber Optic Internet** have a higher churn rate.

**Recommendation**

Investigate pricing, network quality, and customer support for fiber users.
""")

    elif feature == "SeniorCitizen":
        st.info("""
**Business Insight**

Senior citizens show a different churn pattern compared to other customers.

**Recommendation**

Provide dedicated customer support and simplified service plans.
""")

    elif feature == "Partner":
        st.info("""
**Business Insight**

Customers without partners tend to churn more often.

**Recommendation**

Introduce family plans or bundled offers to improve customer retention.
""")

    st.write("---")


with tab4:

    st.subheader("📈 Statistical Analysis")

    st.write(
        """
        Statistical tests help determine whether customer attributes
        have a significant relationship with customer churn.
        """
    )

    # Prepare dataset
    df = raw_df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    # Chi-Square Tests
    chi_square_test(df, "Contract")
    chi_square_test(df, "InternetService")
    chi_square_test(df, "SeniorCitizen")
    chi_square_test(df, "Partner")
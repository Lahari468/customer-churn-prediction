import os
from preprocessing import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, cross_validate
import joblib
import shap


def train_best_model(X_train, y_train):
    os.makedirs("models", exist_ok=True)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            max_depth=10,
            min_samples_split=10,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=5,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
        )
        avg_accuracy = scores["test_accuracy"].mean()
        print(name, "accuracy:", avg_accuracy)

        if avg_accuracy > best_score:
            best_score = avg_accuracy
            best_model = model
            best_name = name

    print("Best model:", best_name)

    best_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(best_model)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(explainer, "models/shap_explainer.pkl")

    return best_model

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_and_preprocess()

    train_best_model(
        X_train,
        y_train
    )
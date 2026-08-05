from src.preprocessing import load_and_preprocess
from src.train_models import train_best_model
from src.evaluate import evaluate_model

print("📥 Loading & preprocessing...")
X_train, X_test, y_train, y_test = load_and_preprocess()

print("🤖 Training...")
model = train_best_model(X_train, y_train)

print("📊 Evaluating...")
evaluate_model(model, X_test, y_test)

print("✅ Done — model retrained on form features only")

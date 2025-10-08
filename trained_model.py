import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# === 1. Load Dataset ===
df = pd.read_csv("alzheimers_prediction_dataset.csv")  # Make sure file is in same folder

# === 2. Encode Categorical Features ===
df_model = df.copy()
label_encoders = {}

for col in df_model.select_dtypes(include='object').columns:
    if col != "Alzheimer’s Diagnosis":
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

# Encode target column
target_encoder = LabelEncoder()
df_model["Alzheimer’s Diagnosis"] = target_encoder.fit_transform(df_model["Alzheimer’s Diagnosis"])

# === 3. Split Data ===
X = df_model.drop("Alzheimer’s Diagnosis", axis=1)
y = df_model["Alzheimer’s Diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 5. Evaluation ===
y_pred = model.predict(X_test)

print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%\n")
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Overfitting check
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"✅ Train Accuracy: {train_acc:.2f}")
print(f"✅ Test Accuracy: {test_acc:.2f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"✅ Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# === 6. Feature Importance Plot ===
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='teal')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# === 7. Save Model & Encoders ===
joblib.dump(model, "alzheimers_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("\n✅ Model and encoders saved successfully!")

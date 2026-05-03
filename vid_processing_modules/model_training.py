import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

csv_path = "feature_sets/horn_training_features_master.csv"

df = pd.read_csv(csv_path)

FEATURES = [
    "peak_match",
    "peak_energy",
    "total_band_energy",
    "concentration",
]

X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y #keep the same ratio because we have way more successes than we do fails
)

model = make_pipeline(
    StandardScaler(), #scale stuff (makes sense)
    LogisticRegression(class_weight="balanced", max_iter=1000) #weight the smaller class so its more balanced
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
pred = (probs > 0.3).astype(int)

# Reconstruct test dataframe with metadata
df_test = df.loc[X_test.index].copy()

df_test["true_label"] = y_test
df_test["pred"] = pred

# Optional: add probabilities (very useful)
df_test["prob"] = model.predict_proba(X_test)[:, 1]

# Print metrics
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Extract errors
fn = df_test[(df_test["true_label"] == 1) & (df_test["pred"] == 0)]
print("\nFalse Negatives:")
print(fn[["video_file", "window_start", "window_end"] + FEATURES])

fp = df_test[(df_test["true_label"] == 0) & (df_test["pred"] == 1)]
print("\nFalse Positives:")
print(fp[["video_file", "window_start", "window_end"] + FEATURES])

# Save for inspection
fn.to_csv("feature_sets/false_negatives.csv", index=False)
fp.to_csv("feature_sets/false_positives.csv", index=False)

joblib.dump(
    model,
    "feature_sets/horn_logistic_model_v2.joblib"
)
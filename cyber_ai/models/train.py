"""
ML Model Training Pipeline for NeuroSent Cyber Threat Intelligence.

Trains three models:
  1. Isolation Forest  — anomaly detection
  2. Random Forest     — threat classification
  3. KMeans            — behavior clustering

Saves all models + scaler + label encoder to models/saved/.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "sample_traffic.csv")
SAVE_DIR = os.path.join(BASE_DIR, "saved")

FEATURE_COLUMNS = [
    "duration", "protocol_type", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "rerror_rate",
    "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_serror_rate",
]


def train_all():
    """Train and save all models."""
    # ── Load data ──
    if not os.path.exists(DATA_PATH):
        print("[!] Data file not found. Run 'python data/generate_data.py' first.")
        sys.exit(1)

    print("[*] Loading training data ...")
    df = pd.read_csv(DATA_PATH)
    print(f"    Loaded {len(df)} rows, {len(df.columns)} columns")

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y_labels = df["label"].values

    # ── Scale features ──
    print("[*] Fitting StandardScaler ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Encode labels ──
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    # ── Train/Test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════
    # MODEL 1 — Isolation Forest (Anomaly Detection)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MODEL 1 — Isolation Forest (Anomaly Detection)")
    print("=" * 60)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_train)

    # Evaluate on test set
    anomaly_preds = iso_forest.predict(X_test)
    n_anomalies = (anomaly_preds == -1).sum()
    anomaly_rate = n_anomalies / len(anomaly_preds) * 100

    # True anomalies are non-normal traffic
    y_test_labels = le.inverse_transform(y_test)
    true_anomalies = (y_test_labels != "normal").sum()
    detected_true = 0
    for i in range(len(y_test)):
        if anomaly_preds[i] == -1 and y_test_labels[i] != "normal":
            detected_true += 1
    anomaly_precision = detected_true / max(n_anomalies, 1) * 100

    print(f"  Anomaly Detection Rate:  {anomaly_rate:.1f}%")
    print(f"  True Anomalies in Test:  {true_anomalies}")
    print(f"  Detected True Anomalies: {detected_true}")
    print(f"  Anomaly Precision:       {anomaly_precision:.1f}%")

    iso_path = os.path.join(SAVE_DIR, "isolation_forest.pkl")
    joblib.dump(iso_forest, iso_path)
    print(f"  Saved -> {iso_path}")

    # ═══════════════════════════════════════════════════════
    # MODEL 2 — Random Forest Classifier (Threat Classification)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MODEL 2 — Random Forest Classifier (Threat Classification)")
    print("=" * 60)

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print("  Classification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=4,
    )
    print(report)

    # Feature importances
    importances = rf_clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    print("  Top 10 Feature Importances:")
    for idx in top_indices:
        print(f"    {FEATURE_COLUMNS[idx]:30s}  {importances[idx]:.4f}")

    rf_path = os.path.join(SAVE_DIR, "random_forest.pkl")
    le_path = os.path.join(SAVE_DIR, "label_encoder.pkl")
    sc_path = os.path.join(SAVE_DIR, "scaler.pkl")
    joblib.dump(rf_clf, rf_path)
    joblib.dump(le, le_path)
    joblib.dump(scaler, sc_path)
    print(f"\n  Saved -> {rf_path}")
    print(f"  Saved -> {le_path}")
    print(f"  Saved -> {sc_path}")

    # ═══════════════════════════════════════════════════════
    # MODEL 3 — KMeans (Behavior Clustering)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MODEL 3 — KMeans (Behavior Clustering)")
    print("=" * 60)

    kmeans = KMeans(
        n_clusters=6,
        random_state=42,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(X_scaled)

    cluster_labels = kmeans.labels_
    print(f"  Cluster distribution:")
    for c in range(6):
        count = (cluster_labels == c).sum()
        pct = count / len(cluster_labels) * 100
        print(f"    Cluster {c}: {count:5d} ({pct:.1f}%)")

    km_path = os.path.join(SAVE_DIR, "kmeans.pkl")
    joblib.dump(kmeans, km_path)
    print(f"\n  Saved -> {km_path}")

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — Summary")
    print("=" * 60)
    print(f"  Classification Accuracy:  {accuracy * 100:.2f}%")
    print(f"  Anomaly Detection Rate:   {anomaly_rate:.1f}%")
    print(f"  Behavior Clusters:        6")
    print(f"  Models saved to:          {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train_all()

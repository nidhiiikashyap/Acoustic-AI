import os
import numpy as np
import pandas as pd
import librosa
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# Custom vibrant color palette
VIBRANT_COLORS = ["#ff6b6b", "#6bafff", "#feca57", "#1dd1a1", "#5f27cd", "#ff9ff3"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#111111",
    "font.size": 12
})


# =========================
# CONFIGURATION
# =========================

TRAIN_DIR = "dataset/train"        # labeled data
UNLABELED_DIR = "dataset/unlabeled"  # uncategorized vocal data

CLASS_NAMES = ["Anxiety", "Depression", "Neutral"]  # labels for training


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(file_path: str, duration: float = 5.0) -> np.ndarray:
    """
    Extract acoustic features from an audio file.

    Features:
        - 13 MFCC means
        - mean pitch (F0)
        - pitch std (jitter proxy)
        - RMS std (shimmer proxy)
        - spectral centroid mean
        - zero-crossing rate mean
    """
    # Load first `duration` seconds
    y, sr = librosa.load(file_path, duration=duration)
    # Normalize signal
    y = librosa.util.normalize(y)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Pitch (F0) using piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]
    if pitch_vals.size > 0:
        pitch_mean = np.mean(pitch_vals)
        pitch_std = np.std(pitch_vals)  # jitter proxy
    else:
        pitch_mean = 0.0
        pitch_std = 0.0

    # RMS (energy) – shimmer proxy using std
    rms = librosa.feature.rms(y=y)[0]
    shimmer_proxy = float(np.std(rms))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    # Concatenate all features into one vector
    features = np.hstack([
        mfcc_mean,
        pitch_mean,
        pitch_std,
        shimmer_proxy,
        centroid_mean,
        zcr_mean
    ])

    return features


# =========================
# LOAD LABELED DATA
# =========================

def load_labeled_data(base_dir: str = TRAIN_DIR):
    """
    Load labeled training data from dataset/train/<CLASS_NAME>/ folders.
    Each audio file is one sample with its corresponding label.
    """
    X = []
    y = []

    for label in CLASS_NAMES:
        folder = os.path.join(base_dir, label)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found for class '{label}': {folder}")
            continue

        for fname in os.listdir(folder):
            if fname.lower().endswith(".wav"):
                file_path = os.path.join(folder, fname)
                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(label)
                    print(f"[INFO] Loaded {file_path} as {label}")
                except Exception as e:
                    print(f"[ERROR] Failed on {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\n[INFO] Loaded labeled data: {X.shape[0]} samples, {X.shape[1] if X.size else 0} features.")
    unique, counts = np.unique(y, return_counts=True)
    print("[INFO] Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt}")

    return X, y


# =========================
# TRAIN MODEL
# =========================

def train_model(X: np.ndarray, y: np.ndarray):
    """
    Train an SVM classifier on the extracted features.
    Returns the fitted model and scaler.
    """
    if X.size == 0 or y.size == 0:
        raise ValueError("Empty training data. Check your dataset/train folders.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # If dataset is very small, skip split and train on all
    if X.shape[0] < len(CLASS_NAMES) * 3:
        print("[WARN] Very small dataset, training and validating on full data.")
        model = SVC(kernel='rbf', probability=True)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        print("\n[INFO] Training-set evaluation (no separate validation set):")
        print("Accuracy:", accuracy_score(y, y_pred))
        print(classification_report(y, y_pred))
        return model, scaler, y, y_pred

    # Otherwise, proper train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print("\n[INFO] Validation-set evaluation:")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    return model, scaler, y_val, y_pred

def predict_one(model, scaler, features: np.ndarray):
    """
    Common prediction helper:
    - scales features
    - gets predict_proba
    - chooses class as argmax(probabilities)
    Returns:
        pred_label, proba (1D array), classes (list)
    """
    feats_scaled = scaler.transform([features])
    proba = model.predict_proba(feats_scaled)[0]
    classes = list(model.classes_)
    idx = int(np.argmax(proba))
    pred_label = classes[idx]
    return pred_label, proba, classes



# =========================
# PREDICT ON UNLABELED DATA
# =========================

def predict_unlabeled(model, scaler, base_dir: str = UNLABELED_DIR):
    """
    Run inference on uncategorized vocal data in dataset/unlabeled.
    Returns a DataFrame with one row per file:
        - file
        - risk_status (At Risk / Not At Risk)
        - predicted_condition (Anxiety / Depression / Neutral)
        - probability columns for each class
    """
    results = []

    if not os.path.isdir(base_dir):
        print(f"[WARN] Unlabeled directory not found: {base_dir}")
        return pd.DataFrame()

    classes = list(model.classes_)

    for fname in os.listdir(base_dir):
        if not fname.lower().endswith(".wav"):
            continue

        file_path = os.path.join(base_dir, fname)

        try:
            feats = extract_features(file_path)
            feats_scaled = scaler.transform([feats])
            pred = model.predict(feats_scaled)[0]
            proba = model.predict_proba(feats_scaled)[0]

            # Risk rule: Anxiety or Depression = At Risk, Neutral = Not At Risk
            risk = "At Risk" if pred in ["Anxiety", "Depression"] else "Not At Risk"

            row = {
                "file": fname,
                "risk_status": risk,
                "predicted_condition": pred
            }

            # Add per-class probabilities
            for cls_name, p in zip(classes, proba):
                row[f"prob_{cls_name.lower()}"] = float(round(p, 3))

            results.append(row)
            print(f"[INFO] Predicted {fname}: {risk}, {pred}")

        except Exception as e:
            print(f"[ERROR] Failed on {file_path}: {e}")

    df = pd.DataFrame(results)
    print(f"\n[INFO] Unlabeled prediction completed: {len(df)} files.")
    return df


# =========================
# PLOTTING
# =========================

def plot_overall_stats(df: pd.DataFrame):
    """
    Plot:
      - Count of At Risk vs Not At Risk
      - Count of each predicted condition
    """
    if df.empty:
        print("[WARN] No results to plot.")
        return

    # Risk status
    plt.figure()
    df["risk_status"].value_counts().plot(kind="bar")
    plt.title("People At Risk vs Not At Risk")
    plt.xlabel("Risk Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Condition distribution
    plt.figure()
    df["predicted_condition"].value_counts().plot(kind="bar")
    plt.title("Predicted Mental Health Conditions")
    plt.xlabel("Condition")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y: np.ndarray):
    if y.size == 0:
        return

    plt.figure(figsize=(7, 5))
    palette = ["#ff6b6b", "#1dd1a1", "#54a0ff"]  # red, green, blue

    ax = sns.countplot(x=y, order=CLASS_NAMES, palette=palette)
    plt.title("Training Data Class Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Class", fontsize=13)
    plt.ylabel("Count", fontsize=13)

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color="#222222", xytext=(0, 10),
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()



def plot_confusion_matrix(y_true, y_pred, labels, title="Evaluation Confusion Matrix"):
    if len(y_true) == 0:
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="PuBuGn",
                xticklabels=labels, yticklabels=labels,
                linewidths=2, linecolor="white")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Predicted", fontsize=13)
    plt.ylabel("Actual", fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_risk_pie(df: pd.DataFrame):
    if df.empty:
        return

    plt.figure(figsize=(6, 6))
    pie_colors = ["#ff6b6b", "#1dd1a1"]  # red = at risk, green = safe

    df["risk_status"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        colors=pie_colors,
        textprops={"fontsize": 12, "color": "#222222"}
    )

    plt.title("Risk Distribution (At Risk vs Not At Risk)", fontsize=16, fontweight="bold")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    # 1. Load labeled training data (Anxiety / Depression / Neutral)
    X, y = load_labeled_data()

    # NEW: show training class distribution
    plot_class_distribution(y)

    # 2. Train SVM model (now returns eval labels & predictions too)
    model, scaler, y_eval, y_pred_eval = train_model(X, y)

    joblib.dump(model, "svm_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("[INFO] Saved svm_model.joblib and scaler.joblib")

    # NEW: show confusion matrix for evaluation
    plot_confusion_matrix(y_eval, y_pred_eval, CLASS_NAMES,
                          title="Evaluation Confusion Matrix")

    # 3. Predict on unlabeled vocal data
    df_results = predict_unlabeled(model, scaler)

    # 4. Show per-person table & graphs
    if not df_results.empty:
        print("\n[INFO] Per-person prediction table:")
        print(df_results)

        df_results.to_csv("results_per_person.csv", index=False)
        print("\n[INFO] Saved results to results_per_person.csv")

        # Existing bar charts
        plot_overall_stats(df_results)

        # NEW: colorful risk pie chart
        plot_risk_pie(df_results)
    else:
        print("[INFO] No unlabeled data processed.")


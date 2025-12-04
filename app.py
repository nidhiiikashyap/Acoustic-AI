import os
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import joblib
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

MODEL_PATH = "svm_model.joblib"
SCALER_PATH = "scaler.joblib"
CLASS_NAMES = ["Anxiety", "Depression", "Neutral"]

EMOJI_MAP = {
    "Anxiety": "😰",
    "Depression": "😔",
    "Neutral": "🙂"
}


# =========================
# LOAD MODEL & SCALER
# =========================

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# =========================
# FEATURE EXTRACTION (EXACTLY LIKE main.py)
# =========================

def extract_features_from_file(file, duration: float = 5.0) -> np.ndarray:
    """
    Same as main.py: extract_features(file_path, duration=5.0)
    Only difference: takes a file-like object instead of path.
    """
    # Load first `duration` seconds
    y, sr = librosa.load(file, duration=duration)
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

def predict_one(model, scaler, features: np.ndarray):
    """
    Same helper as in main.py.
    Ensures Streamlit app and main.py use IDENTICAL prediction logic.
    """
    feats_scaled = scaler.transform([features])
    proba = model.predict_proba(feats_scaled)[0]
    classes = list(model.classes_)
    idx = int(np.argmax(proba))
    pred_label = classes[idx]
    return pred_label, proba, classes


# =========================
# STREAMLIT DASHBOARD
# =========================

def main():
    st.set_page_config(
        page_title="Acoustic AI – Mental Health Screening",
        page_icon="🎧",
        layout="wide"
    )

    # Apple-style heading
    st.markdown(
        """
        <style>
            .apple-title {
                font-size: 52px;
                font-weight: 600;
                text-align: center;
                color: #f5f5f7;
                letter-spacing: -1px;
                margin-top: -20px;
                margin-bottom: 0px;
            }

            .apple-subtitle {
                font-size: 20px;
                font-weight: 300;
                text-align: center;
                color: #d2d2d7;
                margin-top: 5px;
                margin-bottom: 40px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="apple-title">🎧 Acoustic AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="apple-subtitle">Voice-based Mental Health Screening</p>', unsafe_allow_html=True)

    st.markdown(
        """
        <hr style='
            margin-top: -10px;
            margin-bottom: 25px;
            border: 0;
            height: 1px;
            background: linear-gradient(to right, transparent, #555, transparent);
        '>
        """,
        unsafe_allow_html=True
    )

    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.error("Model or scaler file not found. Please run main.py first to train and save them.")
        return

    # Upload section
    top_left, top_right = st.columns([1.4, 1])
    with top_left:
        uploaded_file = st.file_uploader(
            "📂 Upload an audio file (.wav recommended)",
            type=["wav", "mp3", "m4a"]
        )
    with top_right:
        st.info("Tip: Use a clear recording with minimal background noise for best results.")

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        st.markdown("🔄 **Processing audio...**")

        try:
            # 1) Extract features EXACTLY like training
            features = extract_features_from_file(uploaded_file)

            # 2) Use the same helper as main.py
            pred, proba, classes = predict_one(model, scaler, features)

            # 3) Build prob dict for table + graph
            prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}

            # 4) Risk rule identical to main.py
            risk_status = "At Risk" if pred in ["Anxiety", "Depression"] else "Not At Risk"


            # 4) Risk rule (same logic as main.py's predict_unlabeled)
            if pred in ["Anxiety", "Depression"]:
                risk_status = "At Risk"
            else:
                risk_status = "Not At Risk"

            st.markdown("---")
            col1, col2 = st.columns([1, 1.3])

            # LEFT: prediction text
            with col1:
                st.markdown("## 🧠 Prediction Result")

                # Risk badge
                if risk_status == "At Risk":
                    risk_color = "#ff6b6b"
                    risk_text_color = "white"
                else:
                    risk_color = "#1dd1a1"
                    risk_text_color = "white"

                st.markdown(
                    f"**Risk Status:** "
                    f"<span style='background-color:{risk_color};color:{risk_text_color};"
                    f"padding:4px 10px;border-radius:12px;'>"
                    f"{risk_status}</span>",
                    unsafe_allow_html=True
                )

                # Condition badge with emoji
                emoji = EMOJI_MAP.get(pred, "🧩")
                st.markdown(
                    f"**Predicted Condition:** "
                    f"<span style='background-color:#2ecc71;color:white;"
                    f"padding:4px 10px;border-radius:12px;'>"
                    f"{emoji} {pred}</span>",
                    unsafe_allow_html=True
                )

            # RIGHT: probabilities table + bar chart
            with col2:
                st.markdown("## 📊 Class Probabilities")

                prob_rows = []
                for cls in classes:
                    prob_rows.append({
                        "Emotion": f"{EMOJI_MAP.get(cls, '')} {cls}",
                        "Probability (%)": round(prob_dict[cls] * 100, 2),
                    })

                prob_df = pd.DataFrame(prob_rows)
                prob_df.index = prob_df.index + 1  # start at 1

                # Table with color bars
                st.dataframe(
                    prob_df.style.bar(
                        subset=["Probability (%)"],
                        color="#ff6b6b"
                    ),
                    use_container_width=True
                )

                # Horizontal bar chart
                st.markdown("### 📉 Probability Distribution")
                emotions = prob_df["Emotion"]
                values = prob_df["Probability (%)"]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(emotions, values, color="#74b9ff")
                ax.set_xlabel("Probability (%)", fontsize=12)
                ax.set_title("Probability Distribution", fontsize=15, fontweight="bold")
                ax.tick_params(axis='y', labelsize=11)
                ax.invert_yaxis()  # highest on top
                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing audio: {e}")


if __name__ == "__main__":
    main()

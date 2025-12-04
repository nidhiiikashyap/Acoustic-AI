import os
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import joblib

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
# FEATURE EXTRACTION
# =========================

def extract_features_from_audio(y, sr, duration=5.0):
    """
    Extract features from a raw audio signal y, sr.
    We trim or pad to `duration` seconds to keep things consistent.
    """
    target_length = int(duration * sr)
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))

    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]
    if pitch_vals.size > 0:
        pitch_mean = np.mean(pitch_vals)
        pitch_std = np.std(pitch_vals)
    else:
        pitch_mean = 0.0
        pitch_std = 0.0

    rms = librosa.feature.rms(y=y)[0]
    shimmer_proxy = float(np.std(rms))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    features = np.hstack([
        mfcc_mean,
        pitch_mean,
        pitch_std,
        shimmer_proxy,
        centroid_mean,
        zcr_mean
    ])
    return features


def extract_features_from_file(file) -> np.ndarray:
    y, sr = librosa.load(file, sr=None)
    return extract_features_from_audio(y, sr)


# =========================
# STREAMLIT DASHBOARD
# =========================

def main():
    st.set_page_config(
        page_title="Acoustic AI – Mental Health Risk Screening",
        page_icon="🎧",
        layout="wide"
    )

    # HEADER
    st.markdown(
        """
        <style>
            .big-title {
                font-size: 100px;
                font-weight: 700;
            }
            .sub-text {
                font-size: 50px;
                color: #cccccc;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
            .main-title {
                font-size: 100px;
                font-weight: 800;
                text-align: center;
                color: white;
                margin-top: -30px;
                margin-bottom: 10px;
            }
            .sub-text {
                font-size: 50px;
                text-align: center;
                color: #bbbbbb;
                margin-bottom: 30px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="main-title">🎧 Acoustic AI – Voice-based Mental Health Screening</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sub-text">Upload a short speech recording (~30–60 seconds) for vocal-pattern-based mental health analysis.</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-text">Upload a short speech recording (~30–60 seconds). '
        'The system will analyze vocal patterns and estimate risk for anxiety or depression.</p>',
        unsafe_allow_html=True
    )

    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.error("Model or scaler file not found. Please run main.py first to train and save them.")
        return

    st.markdown("---")

    # LAYOUT: audio + upload on top
    left, right = st.columns([1.2, 1])

    with left:
        uploaded_file = st.file_uploader(
            "📂 Upload an audio file (.wav recommended)",
            type=["wav", "mp3", "m4a"]
        )

    with right:
        st.info("Tip: Use a clear recording with minimal background noise for best results.")

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        st.markdown("🔄 **Processing audio...**")

        try:
            features = extract_features_from_file(uploaded_file)
            features_scaled = scaler.transform([features])
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]

            # Risk rule
            risk_status = "At Risk" if pred in ["Anxiety", "Depression"] else "Not At Risk"

            classes = list(model.classes_)
            prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}

            # DASHBOARD LAYOUT
            st.markdown("---")
            col1, col2 = st.columns([1, 1.3])

            # ========= LEFT: TEXT RESULTS =========
            with col1:
                st.markdown("## 🧠 Prediction Result")

                # Risk badge
                if risk_status == "At Risk":
                    st.markdown(
                        f"**Risk Status:** "
                        f"<span style='background-color:#ff6b6b;color:white;padding:4px 10px;border-radius:12px;'>"
                        f"{risk_status}</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"**Risk Status:** "
                        f"<span style='background-color:#1dd1a1;color:white;padding:4px 10px;border-radius:12px;'>"
                        f"{risk_status}</span>",
                        unsafe_allow_html=True
                    )

                # Condition badge with emoji
                emoji = EMOJI_MAP.get(pred, "🧩")
                st.markdown(
                    f"**Predicted Condition:** "
                    f"<span style='background-color:#2ecc71;color:white;padding:4px 10px;border-radius:12px;'>"
                    f"{emoji} {pred}</span>",
                    unsafe_allow_html=True
                )

            # ========= RIGHT: PROBABILITY TABLE + BAR CHART =========
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

                # Color bars in the table
                st.dataframe(
                    prob_df.style.bar(
                        subset=["Probability (%)"],
                        color="#ff6b6b"
                    ),
                    use_container_width=True
                )

                # Horizontal bar chart under the table
                st.markdown("### 📉 Probability Distribution")
                bar_df = prob_df.set_index("Emotion")["Probability (%)"]
                st.bar_chart(bar_df)

        except Exception as e:
            st.error(f"Error processing audio: {e}")


if __name__ == "__main__":
    main()

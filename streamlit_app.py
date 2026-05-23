from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "AUDIO"
CLASS_NAMES = ("healthy", "dementia")
DISPLAY_LABELS = {
    0: "Closer to healthy speech examples",
    1: "Closer to dementia speech examples",
}
RESULT_TONES = {0: "success", 1: "warning"}
SOURCE_HELP = {
    "Upload audio": "Upload a short voice sample in WAV, MP3, M4A, OGG, or FLAC format.",
    "Use project sample": "Pick one of the audio files already available in this project.",
    "Record in browser": "Use your browser microphone to record a short spoken sample.",
}
SUPPORTED_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


@dataclass
class ModelBundle:
    model: RandomForestClassifier
    accuracy: float
    report: str
    sample_count: int
    feature_count: int
    train_count: int
    test_count: int
    class_counts: dict[str, int]
    samples: list[dict[str, object]]
    skipped_files: list[str]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

            /* ── Base ── */
            html, body, .stApp {
                font-family: 'DM Sans', sans-serif !important;
                background-color: #F7F5F0 !important;
                color: #1A1A1A !important;
            }
            .block-container {
                max-width: 1200px;
                padding-top: 2rem !important;
                padding-bottom: 4rem !important;
            }
            * { color: #1A1A1A; }

            /* ── Sidebar ── */
            [data-testid="stSidebar"] {
                background: #FFFFFF !important;
                border-right: 1px solid #E8E3DC !important;
            }
            [data-testid="stSidebar"] * { color: #1A1A1A !important; }
            [data-testid="stSidebar"] .stMetric label { color: #6B6560 !important; font-size: 0.8rem !important; }
            [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] { color: #1A1A1A !important; font-size: 1.6rem !important; font-weight: 700 !important; }

            /* ── Hero Banner ── */
            .hero-wrap {
                background: #FFFFFF;
                border: 1px solid #E8E3DC;
                border-radius: 20px;
                padding: 2.2rem 2.5rem 1.8rem;
                margin-bottom: 1.6rem;
                display: flex;
                align-items: flex-start;
                gap: 2rem;
            }
            .hero-icon {
                width: 60px; height: 60px;
                background: linear-gradient(135deg, #1A7A6A 0%, #24A896 100%);
                border-radius: 16px;
                display: flex; align-items: center; justify-content: center;
                font-size: 28px;
                flex-shrink: 0;
            }
            .hero-title {
                font-family: 'DM Serif Display', serif;
                font-size: 2.1rem;
                font-weight: 400;
                color: #111111;
                line-height: 1.15;
                margin-bottom: 0.4rem;
            }
            .hero-sub {
                font-size: 1rem;
                color: #5A5450;
                line-height: 1.65;
                max-width: 600px;
            }
            .hero-pills {
                display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;
            }
            .hero-pill {
                font-size: 0.82rem; font-weight: 600;
                padding: 0.3rem 0.75rem;
                border-radius: 999px;
                background: #F0EDE8;
                color: #3D3A36;
                border: 1px solid #DDD8D1;
            }
            .hero-pill.green { background: #E3F4EF; color: #1A7A6A; border-color: #B3DDD4; }

            /* ── Stat Cards ── */
            [data-testid="stMetric"] {
                background: #FFFFFF !important;
                border: 1px solid #E8E3DC !important;
                border-radius: 14px !important;
                padding: 1.1rem 1.3rem !important;
            }
            [data-testid="stMetric"] label {
                font-size: 0.78rem !important;
                font-weight: 600 !important;
                letter-spacing: 0.05em !important;
                text-transform: uppercase !important;
                color: #8A8480 !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.8rem !important;
                font-weight: 700 !important;
                color: #111111 !important;
            }

            /* ── Tabs ── */
            .stTabs [data-baseweb="tab-list"] {
                background: transparent !important;
                gap: 0.25rem;
                border-bottom: 1px solid #E8E3DC;
            }
            .stTabs [data-baseweb="tab"] {
                background: transparent !important;
                border: none !important;
                color: #8A8480 !important;
                font-weight: 500 !important;
                font-size: 0.95rem !important;
                padding: 0.65rem 1.1rem !important;
                border-radius: 0 !important;
            }
            .stTabs [aria-selected="true"] {
                color: #1A7A6A !important;
                border-bottom: 2px solid #1A7A6A !important;
            }

            /* ── Section Cards ── */
            .s-card {
                background: #FFFFFF;
                border: 1px solid #E8E3DC;
                border-radius: 16px;
                padding: 1.4rem 1.5rem;
                margin-bottom: 1rem;
            }
            .s-card-title {
                font-size: 0.95rem; font-weight: 700; color: #111111;
                margin-bottom: 0.25rem;
            }
            .s-card-copy {
                font-size: 0.9rem; color: #6B6560; line-height: 1.55;
            }

            /* ── Steps ── */
            .steps-grid { display: grid; gap: 0.9rem; margin-top: 0.85rem; }
            .step-row { display: flex; gap: 1rem; align-items: flex-start; }
            .step-badge {
                width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
                background: #1A7A6A; color: #FFFFFF;
                font-size: 0.85rem; font-weight: 700;
                display: flex; align-items: center; justify-content: center;
            }
            .step-body strong { display: block; font-size: 0.92rem; color: #111111; margin-bottom: 2px; }
            .step-body span { font-size: 0.87rem; color: #6B6560; line-height: 1.45; }

            /* ── Radio Buttons ── */
            .stRadio > div { gap: 0.5rem !important; }
            .stRadio label {
                background: #FFFFFF !important;
                border: 1px solid #DDD8D1 !important;
                border-radius: 10px !important;
                padding: 0.5rem 1rem !important;
                cursor: pointer;
                font-size: 0.9rem !important;
                color: #3D3A36 !important;
                transition: border-color 0.15s;
            }
            .stRadio label:hover { border-color: #1A7A6A !important; }
            .stRadio [aria-checked="true"] + label,
            .stRadio input:checked + div {
                border-color: #1A7A6A !important;
                background: #F0FAF7 !important;
            }

            /* ── File Uploader — deep override ── */
            [data-testid="stFileUploadDropzone"],
            [data-testid="stFileUploadDropzone"] > div,
            .stFileUploader, .stFileUploader > div,
            .stFileUploader section, .stFileUploader section > div {
                background: #FFFFFF !important;
                background-color: #FFFFFF !important;
                color: #3D3A36 !important;
            }
            [data-testid="stFileUploadDropzone"] {
                border: 1.5px dashed #1A7A6A !important;
                border-radius: 12px !important;
                padding: 1rem !important;
            }
            [data-testid="stFileUploadDropzone"] button,
            [data-testid="stFileUploadDropzone"] button span,
            [data-testid="stFileUploadDropzone"] button p {
                background: #1A7A6A !important;
                background-color: #1A7A6A !important;
                color: #FFFFFF !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: 700 !important;
            }
            [data-testid="stFileUploadDropzone"] button:hover {
                background: #155f52 !important;
                background-color: #155f52 !important;
            }
            [data-testid="stFileUploadDropzone"] svg,
            [data-testid="stFileUploadDropzone"] svg * { fill: #1A7A6A !important; }
            [data-testid="stFileUploadDropzone"] small,
            [data-testid="stFileUploadDropzone"] span,
            [data-testid="stFileUploadDropzone"] p { color: #6B6560 !important; }

            /* ── Primary Button ── */
            .stButton > button[kind="primary"],
            .stButton > button[data-testid*="primary"] {
                background: linear-gradient(135deg, #1A7A6A 0%, #24A896 100%) !important;
                border: none !important;
                color: #FFFFFF !important;
                font-weight: 700 !important;
                font-size: 1rem !important;
                border-radius: 12px !important;
                padding: 0.75rem 1.5rem !important;
                letter-spacing: 0.02em;
                transition: opacity 0.2s;
            }
            .stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }

            /* ── Audio Player ── */
            audio { border-radius: 8px; width: 100%; }

            /* ── Result Card ── */
            .r-card {
                background: #FFFFFF;
                border: 1px solid #E8E3DC;
                border-radius: 14px;
                padding: 1.1rem 1.3rem;
            }
            .r-label {
                font-size: 0.75rem; font-weight: 700;
                letter-spacing: 0.06em; text-transform: uppercase;
                color: #8A8480; margin-bottom: 4px;
            }
            .r-value { font-size: 1.55rem; font-weight: 700; color: #111111; }

            /* ── Empty State ── */
            .empty-state {
                background: #FAFAF8;
                border: 1.5px dashed #DDD8D1;
                border-radius: 14px;
                padding: 1.5rem;
                color: #8A8480;
                font-size: 0.9rem;
                line-height: 1.55;
                text-align: center;
            }

            /* ── Alerts ── */
            .stSuccess, .stWarning {
                border-radius: 12px !important;
                border: none !important;
                font-size: 0.93rem !important;
            }

            /* ── Subheadings ── */
            .stMarkdown h3 { font-size: 1rem; font-weight: 700; color: #111111; margin-top: 1.2rem; }

            /* ── Selectbox ── */
            .stSelectbox [data-baseweb="select"] {
                background: #FFFFFF !important;
                border-color: #DDD8D1 !important;
                border-radius: 10px !important;
            }

            /* ── Caption ── */
            .stCaption { font-size: 0.82rem !important; color: #8A8480 !important; }

            /* ── Dataframe ── */
            .stDataFrame { border-radius: 12px !important; overflow: hidden; }

            /* UI refresh overrides */
            [data-testid="stHeader"] {
                background: #F7F5F0 !important;
                box-shadow: none !important;
            }
            [data-testid="stToolbar"],
            #MainMenu,
            footer {
                visibility: hidden !important;
            }
            .block-container {
                max-width: 1180px !important;
                padding-top: 4.25rem !important;
                padding-left: 2rem !important;
                padding-right: 2rem !important;
            }
            .hero-wrap {
                border-radius: 8px !important;
                padding: 1.65rem 1.75rem !important;
                margin-bottom: 1rem !important;
                box-shadow: 0 18px 42px rgba(24, 21, 18, 0.05);
            }
            .hero-icon {
                border-radius: 8px !important;
                color: #FFFFFF !important;
                font-weight: 800 !important;
                font-size: 0.9rem !important;
                letter-spacing: 0.06em !important;
            }
            .hero-title {
                font-size: 2rem !important;
                margin: 0 0 0.35rem !important;
            }
            .hero-sub {
                max-width: 720px !important;
            }
            .s-card,
            .r-card,
            [data-testid="stMetric"] {
                border-radius: 8px !important;
                box-shadow: 0 10px 28px rgba(24, 21, 18, 0.04);
            }
            .s-card {
                padding: 1.35rem 1.45rem !important;
            }
            .empty-state {
                border-radius: 8px !important;
                min-height: 5.4rem;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .stTabs [data-baseweb="tab-list"] {
                margin-top: 0.7rem !important;
            }
            .stTabs [data-baseweb="tab"] p {
                color: inherit !important;
            }
            .stRadio [role="radiogroup"] {
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
            }
            .stRadio [role="radiogroup"] label {
                min-height: 3.05rem !important;
                display: inline-flex !important;
                align-items: center !important;
                gap: 0.45rem !important;
                margin: 0 !important;
            }
            .stRadio [role="radiogroup"] label:has(input:checked) {
                border-color: #1A7A6A !important;
                background: #EAF7F4 !important;
                box-shadow: inset 0 0 0 1px #1A7A6A;
            }
            .stRadio input[type="radio"] {
                accent-color: #1A7A6A !important;
            }
            .stRadio label p,
            .stRadio label span {
                color: #1A1A1A !important;
                font-weight: 600 !important;
            }
            [data-testid="stFileUploader"] label,
            [data-testid="stFileUploader"] label p {
                color: #1A1A1A !important;
                font-weight: 700 !important;
            }
            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploadDropzone"] {
                background: #FFFFFF !important;
                border: 1px dashed #BFB7AD !important;
                border-radius: 8px !important;
                padding: 1rem !important;
            }
            [data-testid="stFileUploader"] section *,
            [data-testid="stFileUploadDropzone"] * {
                color: #5A5450 !important;
            }
            [data-testid="stFileUploader"] button,
            [data-testid="stFileUploadDropzone"] button {
                background: #1A7A6A !important;
                border: 1px solid #1A7A6A !important;
                border-radius: 8px !important;
                color: #FFFFFF !important;
                min-height: 2.6rem !important;
                padding: 0.45rem 1rem !important;
            }
            [data-testid="stFileUploader"] button *,
            [data-testid="stFileUploadDropzone"] button * {
                color: #FFFFFF !important;
                font-weight: 700 !important;
            }
            .stButton > button {
                border-radius: 8px !important;
                min-height: 3rem !important;
            }
            .stButton > button[kind="primary"],
            .stButton > button[data-testid*="primary"],
            [data-testid="stBaseButton-primary"] {
                background: #1A7A6A !important;
                border: 1px solid #1A7A6A !important;
                color: #FFFFFF !important;
                box-shadow: 0 12px 24px rgba(26, 122, 106, 0.2);
            }
            .stButton > button[kind="primary"] *,
            .stButton > button[data-testid*="primary"] *,
            [data-testid="stBaseButton-primary"] * {
                color: #FFFFFF !important;
            }
            [data-testid="stSidebar"] {
                min-width: 320px !important;
            }
            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 1rem !important;
            }
            @media (max-width: 900px) {
                .block-container {
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;
                    padding-top: 3.75rem !important;
                }
                .hero-wrap {
                    flex-direction: column;
                    gap: 1rem;
                }
                .hero-title {
                    font-size: 1.6rem !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_features(file_path: str | Path) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    if y.size == 0:
        raise ValueError("The selected audio file is empty.")
    harmonic = librosa.effects.harmonic(y)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    tonnetz = np.mean(librosa.feature.tonnetz(y=harmonic, sr=sr), axis=1)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz]).astype(np.float32)


def load_audio_overview(file_path: str | Path) -> tuple[np.ndarray, int, dict[str, float]]:
    y, sr = librosa.load(file_path, sr=None)
    stats = {
        "duration_seconds": float(librosa.get_duration(y=y, sr=sr)),
        "mean_rms_energy": float(np.mean(librosa.feature.rms(y=y))),
        "mean_zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
    }
    return y, sr, stats


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[str]]:
    X: list[np.ndarray] = []
    y: list[int] = []
    samples: list[dict[str, object]] = []
    skipped_files: list[str] = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            skipped_files.append(f"Missing folder: {class_dir}")
            continue
        for file_path in sorted(class_dir.glob("*.wav")):
            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
                samples.append({
                    "label": class_name,
                    "file_name": file_path.name,
                    "path": str(file_path.relative_to(BASE_DIR)),
                })
            except Exception as exc:
                skipped_files.append(f"{file_path.name}: {exc}")

    if not X:
        raise RuntimeError(
            "No training audio could be loaded. Add WAV files to AUDIO/healthy "
            "and AUDIO/dementia first."
        )
    return np.array(X), np.array(y), samples, skipped_files


def train_model(dataset_dir: str) -> ModelBundle:
    dataset_path = Path(dataset_dir)
    X, y, samples, skipped_files = load_dataset(dataset_path)
    class_counts = {
        class_name: int(np.sum(y == label))
        for label, class_name in enumerate(CLASS_NAMES)
    }
    if min(class_counts.values()) < 2:
        raise RuntimeError(
            "Each class needs at least two audio files so the app can validate the model."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    evaluation_model = RandomForestClassifier(n_estimators=200, random_state=42)
    evaluation_model.fit(X_train, y_train)
    y_pred = evaluation_model.predict(X_test)
    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X, y)
    report = classification_report(
        y_test, y_pred,
        labels=[0, 1], target_names=list(CLASS_NAMES), zero_division=0,
    )
    return ModelBundle(
        model=final_model,
        accuracy=float(accuracy_score(y_test, y_pred)),
        report=report,
        sample_count=int(len(X)),
        feature_count=int(X.shape[1]),
        train_count=int(len(X_train)),
        test_count=int(len(X_test)),
        class_counts=class_counts,
        samples=samples,
        skipped_files=skipped_files,
    )


def list_project_audio_files() -> dict[str, Path]:
    choices: dict[str, Path] = {}
    for file_path in sorted(BASE_DIR.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            choices[f"Project file / {file_path.name}"] = file_path
    for class_name in CLASS_NAMES:
        class_dir = DATASET_DIR / class_name
        for file_path in sorted(class_dir.glob("*.wav")):
            choices[f"Dataset {class_name} / {file_path.name}"] = file_path
    return choices


def persist_uploaded_audio(audio_bytes: bytes, suffix: str) -> Path:
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=normalized_suffix) as tmp:
        tmp.write(audio_bytes)
        return Path(tmp.name)


def predict_audio(model_bundle: ModelBundle, file_path: Path) -> dict[str, object]:
    y, sr, stats = load_audio_overview(file_path)
    features = extract_features(file_path)
    prediction = int(model_bundle.model.predict([features])[0])
    probabilities = model_bundle.model.predict_proba([features])[0]
    return {
        "prediction": prediction,
        "label": DISPLAY_LABELS[prediction],
        "confidence": float(probabilities[prediction]),
        "probabilities": {
            "healthy": float(probabilities[0]),
            "dementia": float(probabilities[1]),
        },
        "waveform": y,
        "sample_rate": sr,
        "stats": stats,
        "feature_vector": features,
    }


def render_waveform(y: np.ndarray, sr: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    timeline = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(timeline, y, color="#1A7A6A", linewidth=0.9)
    ax.fill_between(timeline, y, 0, color="#1A7A6A", alpha=0.12)
    ax.set_title("Waveform", fontsize=11, color="#3D3A36", pad=8)
    ax.set_xlabel("Time (s)", fontsize=10, color="#6B6560")
    ax.set_ylabel("Amplitude", fontsize=10, color="#6B6560")
    ax.tick_params(colors="#8A8480", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#E8E3DC")
    ax.grid(color="#F0EDE8", linewidth=0.7)
    fig.tight_layout()
    return fig


def card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="s-card"><div class="s-card-title">{title}</div>'
        f'<div class="s-card-copy">{body}</div></div>',
        unsafe_allow_html=True,
    )


def result_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="r-card"><div class="r-label">{label}</div>'
        f'<div class="r-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_steps() -> None:
    st.markdown(
        """
        <div class="s-card">
            <div class="s-card-title">How to use</div>
            <div class="steps-grid" style="margin-top:0.9rem">
                <div class="step-row">
                    <div class="step-badge">1</div>
                    <div class="step-body">
                        <strong>Choose an input method</strong>
                        <span>Upload a file, pick a project sample, or record in your browser.</span>
                    </div>
                </div>
                <div class="step-row">
                    <div class="step-badge">2</div>
                    <div class="step-body">
                        <strong>Preview the audio</strong>
                        <span>Listen once before analysis so you know the correct clip is selected.</span>
                    </div>
                </div>
                <div class="step-row">
                    <div class="step-badge">3</div>
                    <div class="step-body">
                        <strong>Read the result summary</strong>
                        <span>Review the model label, confidence, and waveform. Use it as a demo signal, not a diagnosis.</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_interpretation(prediction: int, confidence: float) -> None:
    msg = (
        "This sample is currently closer to the **healthy** examples in the training dataset."
        if prediction == 0
        else "This sample is currently closer to the **dementia** examples in the training dataset."
    )
    note = (
        " The model confidence is fairly strong for this sample."
        if confidence >= 0.75
        else " The model confidence is moderate — treat this result with extra caution."
    )
    full = msg + note
    if RESULT_TONES[prediction] == "success":
        st.success(full)
    else:
        st.warning(full)


def main() -> None:
    st.set_page_config(
        page_title="Dementia Speech Screening Demo",
        page_icon=":material/record_voice_over:",
        layout="wide",
    )
    inject_styles()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-icon">AI</div>
            <div>
                <div class="hero-title">Dementia Speech Screening</div>
                <div class="hero-sub">
                    A voice-analysis demo built on your local dataset. Select a sample,
                    run the analysis, and read the result in plain language.
                </div>
                <div class="hero-pills">
                    <span class="hero-pill green">Research demo only</span>
                    <span class="hero-pill">Audio preview</span>
                    <span class="hero-pill">Waveform analysis</span>
                    <span class="hero-pill">Confidence score</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Important: Not a medical device. Do not use as a diagnosis.")

    if not DATASET_DIR.exists():
        st.error(f"Dataset folder not found: {DATASET_DIR}")
        st.stop()

    try:
        cached_train = st.cache_resource(
            show_spinner="Training model from local audio dataset…"
        )(train_model)
        bundle = cached_train(str(DATASET_DIR))
    except Exception as exc:
        st.error(f"Could not prepare the model: {exc}")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Overview")
        st.write("Trained on audio samples in the project directory.")
        st.metric("Samples", bundle.sample_count)
        st.metric("Validation accuracy", f"{bundle.accuracy:.1%}")
        st.caption("This result shows which group the sample is closer to. Not a diagnosis.")

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Dataset samples", bundle.sample_count)
    with c2: st.metric("Validation accuracy", f"{bundle.accuracy:.1%}")
    with c3: st.metric("Healthy samples", bundle.class_counts["healthy"])
    with c4: st.metric("Dementia samples", bundle.class_counts["dementia"])

    st.session_state.setdefault("analysis_result", None)

    analyze_tab, dataset_tab, details_tab = st.tabs(
        ["Analyze Audio", "Dataset", "Model Details"]
    )

    # ── Analyze Tab ───────────────────────────────────────────────────────────
    with analyze_tab:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        input_col, result_col = st.columns([1, 1], gap="large")

        with input_col:
            card(
                "Source Selection",
                "Choose one audio source below, preview it, then click Analyze Audio.",
            )

            source = st.radio(
                "Choose an input source",
                options=["Upload audio", "Use project sample", "Record in browser"],
                horizontal=True,
                label_visibility="collapsed",
            )
            st.caption(SOURCE_HELP[source])
            st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)

            selected_path: Path | None = None
            audio_bytes: bytes | None = None
            selected_name = "No audio selected"

            if source == "Upload audio":
                uploaded = st.file_uploader(
                    "Upload a voice sample",
                    type=["wav", "mp3", "m4a", "ogg", "flac"],
                    help="Short, clear speech clips work best.",
                )
                if uploaded is not None:
                    audio_bytes = uploaded.getvalue()
                    suffix = Path(uploaded.name).suffix or ".wav"
                    selected_path = persist_uploaded_audio(audio_bytes, suffix)
                    selected_name = uploaded.name

            elif source == "Use project sample":
                options = list_project_audio_files()
                if not options:
                    st.info("No sample audio files were found in the project.")
                else:
                    sel = st.selectbox("Choose a sample file", list(options.keys()))
                    selected_path = options[sel]
                    audio_bytes = selected_path.read_bytes()
                    selected_name = selected_path.name
            else:
                if hasattr(st, "audio_input"):
                    recorded = st.audio_input("Record a short speech sample")
                    if recorded is not None:
                        audio_bytes = recorded.getvalue()
                        selected_path = persist_uploaded_audio(audio_bytes, ".wav")
                        selected_name = "Browser microphone recording"
                else:
                    st.info(
                        "This Streamlit version does not support browser microphone input. "
                        "Use Upload audio instead."
                    )

            if audio_bytes is not None:
                card("Selected Audio", f"Current file: <strong>{selected_name}</strong>")
                st.audio(audio_bytes)
            else:
                st.markdown(
                    '<div class="empty-state">Select or record an audio sample to preview it here before analysis.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            analyze_clicked = st.button(
                "Analyze Audio",
                type="primary",
                use_container_width=True,
            )

            if analyze_clicked:
                if selected_path is None:
                    st.warning("Choose or record an audio sample first.")
                else:
                    try:
                        st.session_state.analysis_result = predict_audio(bundle, selected_path)
                    except Exception as exc:
                        st.session_state.analysis_result = None
                        st.error(f"Audio analysis failed: {exc}")

        # ── Result column ──────────────────────────────────────────────────────
        with result_col:
            card(
                "Result Summary",
                "After analysis, the app explains which training group the sample is closer to and how confident the model is.",
            )

            result = st.session_state.analysis_result
            if result is None:
                st.markdown(
                    '<div class="empty-state">No analysis yet. Choose an audio sample on the left, '
                    'then click <strong>Analyze Audio</strong> to view the result.</div>',
                    unsafe_allow_html=True,
                )
            else:
                r1, r2 = st.columns([1.2, 0.8])
                with r1:
                    result_card("Prediction", result["label"])
                with r2:
                    result_card("Confidence", f"{result['confidence']:.1%}")

                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                render_interpretation(result["prediction"], result["confidence"])

                st.markdown("##### Class probabilities")
                prob_df = pd.DataFrame(
                    {
                        "Class": ["Healthy", "Dementia"],
                        "Probability": [
                            result["probabilities"]["healthy"],
                            result["probabilities"]["dementia"],
                        ],
                    }
                ).set_index("Class")
                st.bar_chart(prob_df, color="#1A7A6A")

                st.markdown("##### Audio snapshot")
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.metric("Duration", f"{result['stats']['duration_seconds']:.2f} s")
                with s2:
                    st.metric("RMS energy", f"{result['stats']['mean_rms_energy']:.4f}")
                with s3:
                    st.metric("Zero-crossing rate", f"{result['stats']['mean_zero_crossing_rate']:.4f}")

                st.markdown("##### Waveform")
                st.pyplot(render_waveform(result["waveform"], result["sample_rate"]))

                with st.expander("Advanced feature preview"):
                    feat_df = pd.DataFrame(
                        {"feature_index": range(len(result["feature_vector"])),
                         "value": result["feature_vector"]}
                    )
                    st.dataframe(feat_df, use_container_width=True, height=280)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        render_steps()

    # ── Dataset Tab ───────────────────────────────────────────────────────────
    with dataset_tab:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        card(
            "Training Dataset",
            "These files are used to train the demo model. Adding more balanced examples usually improves stability.",
        )
        ds_df = pd.DataFrame(bundle.samples).rename(
            columns={"label": "Class", "file_name": "File Name", "path": "Project Path"}
        )
        st.dataframe(ds_df, use_container_width=True, height=360)

        count_df = pd.DataFrame(
            {"Class": ["Healthy", "Dementia"],
             "Samples": [bundle.class_counts["healthy"], bundle.class_counts["dementia"]]}
        ).set_index("Class")
        st.markdown("##### Class balance")
        st.bar_chart(count_df, color="#1A7A6A")

    # ── Details Tab ───────────────────────────────────────────────────────────
    with details_tab:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        card(
            "Model Details",
            "librosa feature extraction + Random Forest classifier trained from the two local dataset folders.",
        )
        d1, d2, d3 = st.columns(3)
        with d1: st.metric("Feature vector length", bundle.feature_count)
        with d2: st.metric("Training split", bundle.train_count)
        with d3: st.metric("Test split", bundle.test_count)

        st.markdown("##### Classification report")
        st.code(bundle.report, language="text")

        if bundle.skipped_files:
            st.warning("Some files were skipped while building the dataset.")
            st.code("\n".join(bundle.skipped_files), language="text")


if __name__ == "__main__":
    main()

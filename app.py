import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ.setdefault("DISPLAY", "")

import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
import tempfile
import base64
import json
from pathlib import Path
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitchGuard AI · Injury Risk Analyser",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0a0f0d; color: #e8f0eb; }
  .stSidebar { background: #0d1410 !important; border-right: 1px solid #1e3a2a; }
  .hero-header { text-align:center; padding:2.5rem 0 1.5rem; border-bottom:1px solid #1e3a2a; margin-bottom:2rem; }
  .hero-title { font-family:'Bebas Neue',cursive; font-size:4rem; letter-spacing:0.12em; color:#fff; line-height:1; margin:0; }
  .hero-accent { color:#4cff8f; }
  .hero-sub { font-size:0.9rem; color:#6b8f78; letter-spacing:0.25em; text-transform:uppercase; margin-top:0.5rem; }
  [data-testid="stFileUploader"] { border:2px dashed #1e3a2a; border-radius:12px; padding:2rem; background:#0d1a12; }
  .risk-card { background:linear-gradient(135deg,#0d1a12 0%,#112018 100%); border:1px solid #1e3a2a; border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1rem; }
  .risk-label { font-size:0.7rem; letter-spacing:0.2em; text-transform:uppercase; color:#6b8f78; margin-bottom:0.3rem; }
  .risk-value { font-family:'Bebas Neue',cursive; font-size:2.6rem; line-height:1; color:#fff; }
  .risk-low { color:#4cff8f; } .risk-med { color:#ffd74c; } .risk-high { color:#ff5c5c; }
  .risk-bar-wrap { background:#0d1a12; border:1px solid #1e3a2a; border-radius:10px; padding:1.2rem 1.6rem; margin-bottom:1rem; }
  .risk-bar-bg { background:#1e2e25; border-radius:6px; height:14px; overflow:hidden; margin-top:0.5rem; }
  .risk-bar-fill { height:100%; border-radius:6px; }
  .section-heading { font-family:'Bebas Neue',cursive; font-size:1.5rem; letter-spacing:0.1em; color:#fff; border-left:4px solid #4cff8f; padding-left:0.75rem; margin:1.5rem 0 1rem; }
  .coach-box { background:#0d1a12; border:1px solid #1e3a2a; border-radius:12px; padding:1.6rem; line-height:1.8; color:#c8ddd0; font-size:0.95rem; white-space:pre-wrap; }
  .frame-caption { font-size:0.72rem; color:#6b8f78; text-align:center; margin-top:0.3rem; letter-spacing:0.1em; text-transform:uppercase; }
  .stButton > button { background:#4cff8f !important; color:#0a0f0d !important; font-weight:600 !important; border:none !important; border-radius:8px !important; padding:0.65rem 2rem !important; }
  .sidebar-label { font-size:0.72rem; letter-spacing:0.15em; text-transform:uppercase; color:#6b8f78; margin-bottom:0.25rem; }
  hr { border-color:#1e3a2a !important; }
  .tag { display:inline-block; background:#0d2a1a; border:1px solid #1e4a2e; border-radius:20px; padding:0.2rem 0.8rem; font-size:0.75rem; color:#4cff8f; margin:0.15rem; }
</style>
""", unsafe_allow_html=True)


def extract_frames(video_path: str, max_frames: int = 10) -> list:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


def annotate_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 32), (w, h), (10, 20, 15), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, f"FRAME {frame_idx + 1:02d}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (76, 255, 143), 1, cv2.LINE_AA)
    return img


def frame_to_b64(frame: np.ndarray) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


SYSTEM_PROMPT = """You are PitchGuard — an elite MLB pitching biomechanics coach and
sports medicine expert. Analyse the sequential pitching frames provided.

Produce a structured report with EXACTLY these sections:

## PITCHING MECHANICS BREAKDOWN
Analyse each phase: Wind-Up, Stride, Arm Cocking, Arm Acceleration, Release, Follow-Through.
Comment on body alignment, balance, hip/shoulder separation, arm path, foot placement.

## INJURY RISK FACTORS
List specific biomechanical red flags (inverted-W arm, early trunk rotation, hyperextended
elbow, collapsed front knee, etc.). For each, explain WHY it raises injury risk.

## INJURY RISK INDEX
Single integer 0 (no risk) to 100 (extreme risk) on its own line:
RISK_INDEX: <integer>

## BODY PART RISK BREAKDOWN
For each body part give: Low / Medium / High + one sentence.
- Elbow (UCL, medial epicondyle)
- Shoulder (rotator cuff, labrum)
- Lower back / spine
- Hip & pelvis
- Knee (front leg)

## COACHING RECOMMENDATIONS
4-6 specific, actionable drill/correction cues to work on immediately.

## SUMMARY
Two-sentence overall assessment.

Be direct, clinical, use correct anatomical terminology."""


def call_gemini(api_key: str, frames_b64: list) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": b}} for b in frames_b64]
    parts.append({"text": SYSTEM_PROMPT})
    response = model.generate_content(parts)
    return response.text


def parse_risk_index(text: str) -> int:
    for line in text.splitlines():
        if "RISK_INDEX:" in line:
            digits = "".join(filter(str.isdigit, line.split("RISK_INDEX:")[-1]))
            try:
                return min(100, max(0, int(digits)))
            except ValueError:
                pass
    return 50


def risk_colour(s): return "#4cff8f" if s < 35 else "#ffd74c" if s < 65 else "#ff5c5c"
def risk_label(s):  return "LOW"     if s < 35 else "MODERATE" if s < 65 else "HIGH"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚾ PitchGuard AI")
    st.markdown("---")
    st.markdown('<p class="sidebar-label">Gemini API Key</p>', unsafe_allow_html=True)
    api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...", label_visibility="collapsed", key="gemini_api_key")
    st.markdown("---")
    st.markdown('<p class="sidebar-label">Analysis Settings</p>', unsafe_allow_html=True)
    max_frames = st.slider("Frames to Extract", min_value=6, max_value=20, value=10)
    st.markdown("---")
    st.markdown("""<small style="color:#6b8f78;line-height:1.7">
PitchGuard sends video frames to <b style="color:#4cff8f">Gemini 1.5 Flash</b>
for full biomechanical analysis, injury risk scoring, and coaching recommendations.
</small>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div><span class="tag">Gemini 1.5 Flash</span> <span class="tag">OpenCV</span> <span class="tag">Streamlit</span></div>', unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">PITCH<span class="hero-accent">GUARD</span> AI</div>
  <div class="hero-sub">Baseball Pitcher · Biomechanics Coach · Injury Risk Analyser</div>
</div>""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Pitching Video", type=["mp4", "mov", "avi", "mkv"],
                             help="Upload a short clip (10–30 s) of the pitcher's full delivery.")

if uploaded:
    st.video(uploaded)
    st.markdown("---")

    if not api_key:
        st.warning("⚠️  Enter your Gemini API key in the sidebar to run the analysis.")
    else:
        if st.button("🔍  Analyse Pitch — Detect Injury Risk"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Extracting frames from video…"):
                    raw_frames = extract_frames(tmp_path, max_frames=max_frames)

                if not raw_frames:
                    st.error("Could not extract frames. Try a different video file.")
                    st.stop()

                st.success(f"✅  Extracted {len(raw_frames)} frames")

                with st.spinner("Preparing frames…"):
                    annotated = [annotate_frame(f, i) for i, f in enumerate(raw_frames)]

                st.markdown('<div class="section-heading">VIDEO FRAMES</div>', unsafe_allow_html=True)
                cols = st.columns(min(5, len(annotated)))
                for i, frame in enumerate(annotated):
                    with cols[i % len(cols)]:
                        st.image(frame, use_column_width=True)
                        st.markdown(f'<div class="frame-caption">Frame {i+1}</div>', unsafe_allow_html=True)

                st.markdown("---")

                with st.spinner("Consulting PitchGuard AI coach (Gemini 1.5 Flash)…"):
                    frames_b64 = [frame_to_b64(f) for f in annotated]
                    analysis   = call_gemini(api_key, frames_b64)

                risk_score = parse_risk_index(analysis)
                colour     = risk_colour(risk_score)
                label      = risk_label(risk_score)

                st.markdown('<div class="section-heading">INJURY RISK INDEX</div>', unsafe_allow_html=True)
                _, c2, _ = st.columns([1, 2, 1])
                with c2:
                    st.markdown(f"""
<div class="risk-card" style="text-align:center;border-color:{colour}40;">
  <div class="risk-label">Overall Injury Risk Score</div>
  <div class="risk-value" style="font-size:5rem;color:{colour};">{risk_score}</div>
  <div style="font-family:'Bebas Neue',cursive;font-size:1.4rem;letter-spacing:0.2em;color:{colour};margin-top:0.25rem;">{label} RISK</div>
  <div class="risk-bar-bg" style="margin-top:1rem;">
    <div class="risk-bar-fill" style="width:{risk_score}%;background:{colour};"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:0.3rem;font-size:0.7rem;color:#6b8f78;">
    <span>0 — Safe</span><span>50 — Moderate</span><span>100 — Critical</span>
  </div>
</div>""", unsafe_allow_html=True)

                lc1, lc2, lc3 = st.columns(3)
                lc1.markdown("""<div class="risk-bar-wrap"><div class="risk-label">Low Risk Zone</div>
<div class="risk-value risk-low">0 – 34</div>
<div class="risk-bar-bg"><div class="risk-bar-fill" style="width:34%;background:#4cff8f;"></div></div></div>""", unsafe_allow_html=True)
                lc2.markdown("""<div class="risk-bar-wrap"><div class="risk-label">Moderate Risk Zone</div>
<div class="risk-value risk-med">35 – 64</div>
<div class="risk-bar-bg"><div class="risk-bar-fill" style="width:64%;background:#ffd74c;"></div></div></div>""", unsafe_allow_html=True)
                lc3.markdown("""<div class="risk-bar-wrap"><div class="risk-label">High Risk Zone</div>
<div class="risk-value risk-high">65 – 100</div>
<div class="risk-bar-bg"><div class="risk-bar-fill" style="width:100%;background:#ff5c5c;"></div></div></div>""", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown('<div class="section-heading">AI COACH ANALYSIS</div>', unsafe_allow_html=True)
                clean = "\n".join(l for l in analysis.splitlines() if "RISK_INDEX:" not in l)
                st.markdown(f'<div class="coach-box">{clean}</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("""<div style="text-align:center;color:#3a6650;font-size:0.75rem;
letter-spacing:0.15em;text-transform:uppercase;padding:1rem 0;">
  PitchGuard AI · Not a substitute for professional medical advice · For coaching use only
</div>""", unsafe_allow_html=True)

            finally:
                os.unlink(tmp_path)

else:
    st.markdown("""
<div style="text-align:center;padding:4rem 0;color:#3a6650;">
  <div style="font-size:5rem;margin-bottom:1rem;">⚾</div>
  <div style="font-family:'Bebas Neue',cursive;font-size:2rem;letter-spacing:0.1em;color:#4a7a60;">
    UPLOAD A PITCHING VIDEO TO BEGIN
  </div>
  <div style="font-size:0.85rem;margin-top:0.5rem;letter-spacing:0.1em;text-transform:uppercase;">
    Supports MP4 · MOV · AVI · MKV
  </div>
</div>""", unsafe_allow_html=True)

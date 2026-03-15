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
import time
from pathlib import Path
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitchGuard AI · Injury Risk Analyser",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark diamond-green theme */
  .stApp {
    background: #0a0f0d;
    color: #e8f0eb;
  }

  .stSidebar {
    background: #0d1410 !important;
    border-right: 1px solid #1e3a2a;
  }

  /* Hero header */
  .hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1e3a2a;
    margin-bottom: 2rem;
  }
  .hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 4rem;
    letter-spacing: 0.12em;
    color: #ffffff;
    line-height: 1;
    margin: 0;
  }
  .hero-accent {
    color: #4cff8f;
  }
  .hero-sub {
    font-size: 0.9rem;
    color: #6b8f78;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 0.5rem;
  }

  /* Upload zone */
  [data-testid="stFileUploader"] {
    border: 2px dashed #1e3a2a;
    border-radius: 12px;
    padding: 2rem;
    background: #0d1a12;
    transition: border-color 0.3s;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #4cff8f;
  }

  /* Metric cards */
  .risk-card {
    background: linear-gradient(135deg, #0d1a12 0%, #112018 100%);
    border: 1px solid #1e3a2a;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
  }
  .risk-label {
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b8f78;
    margin-bottom: 0.3rem;
  }
  .risk-value {
    font-family: 'Bebas Neue', cursive;
    font-size: 2.6rem;
    line-height: 1;
    color: #ffffff;
  }
  .risk-low   { color: #4cff8f; }
  .risk-med   { color: #ffd74c; }
  .risk-high  { color: #ff5c5c; }

  /* Risk bar */
  .risk-bar-wrap {
    background: #0d1a12;
    border: 1px solid #1e3a2a;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1rem;
  }
  .risk-bar-bg {
    background: #1e2e25;
    border-radius: 6px;
    height: 14px;
    overflow: hidden;
    margin-top: 0.5rem;
  }
  .risk-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 1s ease;
  }

  /* Section headings */
  .section-heading {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.5rem;
    letter-spacing: 0.1em;
    color: #ffffff;
    border-left: 4px solid #4cff8f;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
  }

  /* Coach analysis box */
  .coach-box {
    background: #0d1a12;
    border: 1px solid #1e3a2a;
    border-radius: 12px;
    padding: 1.6rem;
    line-height: 1.8;
    color: #c8ddd0;
    font-size: 0.95rem;
    white-space: pre-wrap;
  }

  /* Frame grid */
  .frame-caption {
    font-size: 0.72rem;
    color: #6b8f78;
    text-align: center;
    margin-top: 0.3rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  /* Buttons */
  .stButton > button {
    background: #4cff8f !important;
    color: #0a0f0d !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.05em;
    transition: opacity 0.2s;
  }
  .stButton > button:hover {
    opacity: 0.85 !important;
  }

  /* Sidebar labels */
  .sidebar-label {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b8f78;
    margin-bottom: 0.25rem;
  }

  /* Divider */
  hr { border-color: #1e3a2a !important; }

  /* Info tags */
  .tag {
    display: inline-block;
    background: #0d2a1a;
    border: 1px solid #1e4a2e;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    color: #4cff8f;
    margin: 0.15rem;
    letter-spacing: 0.08em;
  }
</style>
""", unsafe_allow_html=True)

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ── Helper functions ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, max_frames: int = 12) -> list[np.ndarray]:
    """Extract evenly-spaced frames from a video file."""
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step   = max(1, total // max_frames)
    frames = []
    idx    = 0
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


def draw_skeleton(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run MediaPipe Pose on a frame and return annotated image + landmark data."""
    annotated = frame.copy()
    landmark_data = {}

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.4,
    ) as pose:
        results = pose.process(frame)

    if results.pose_landmarks:
        # Draw skeleton overlay
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(76, 255, 143), thickness=2, circle_radius=3
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255), thickness=2
            ),
        )
        for name, lm in zip(mp_pose.PoseLandmark, results.pose_landmarks.landmark):
            landmark_data[name.name] = {
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "visibility": round(lm.visibility, 3),
            }
    return annotated, landmark_data


def frame_to_b64(frame: np.ndarray) -> str:
    """Convert numpy RGB frame to base64 JPEG string."""
    img    = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_gemini_prompt(landmark_summaries: list[dict], num_frames: int) -> str:
    return f"""
You are PitchGuard — an elite MLB pitching biomechanics coach and sports medicine expert.
You have been given {num_frames} sequential frames extracted from a pitcher's delivery video,
along with MediaPipe skeleton landmark data for each frame.

Landmark data (frame-by-frame):
{json.dumps(landmark_summaries, indent=2)}

Your task — analyse the pitcher's mechanics and produce a structured report with EXACTLY these sections:

## PITCHING MECHANICS BREAKDOWN
Analyse each major phase: Wind-Up, Stride, Arm Cocking, Arm Acceleration, Release, Follow-Through.
For each phase, comment on body alignment, balance, hip/shoulder separation, arm path, and foot placement.

## INJURY RISK FACTORS
List specific biomechanical red flags observed (e.g. inverted-W arm position, early trunk rotation,
hyperextended elbow, collapsed front knee, etc.). For each, explain WHY it increases injury risk.

## INJURY RISK INDEX
Provide a single integer score from 0 (no risk) to 100 (extreme risk).
Return it on its own line in this exact format:
RISK_INDEX: <integer>

## BODY PART RISK BREAKDOWN
For each of these body parts, give a brief risk assessment (Low / Medium / High + 1 sentence):
- Elbow (UCL, medial epicondyle)
- Shoulder (rotator cuff, labrum)
- Lower back / spine
- Hip & pelvis
- Knee (front leg)

## COACHING RECOMMENDATIONS
Give 4–6 specific, actionable drill/correction cues the pitcher should work on immediately.

## SUMMARY
Two-sentence overall take for the pitcher.

Be direct, clinical, and use correct anatomical terminology. Do not be vague.
"""


def call_gemini(api_key: str, frames_b64: list[str], landmark_summaries: list[dict]) -> str:
    """Send frames + prompt to Gemini Vision and return the text response."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Build multipart content: interleave frame images with the text prompt
    content_parts = []
    for i, b64 in enumerate(frames_b64):
        content_parts.append({
            "inline_data": {"mime_type": "image/jpeg", "data": b64}
        })
    content_parts.append({"text": build_gemini_prompt(landmark_summaries, len(frames_b64))})

    response = model.generate_content(content_parts)
    return response.text


def parse_risk_index(text: str) -> int:
    """Extract RISK_INDEX integer from Gemini response."""
    for line in text.splitlines():
        if "RISK_INDEX:" in line:
            try:
                return int("".join(filter(str.isdigit, line.split("RISK_INDEX:")[-1])))
            except ValueError:
                pass
    return 50  # fallback


def risk_colour(score: int) -> str:
    if score < 35:
        return "#4cff8f"   # green
    elif score < 65:
        return "#ffd74c"   # amber
    else:
        return "#ff5c5c"   # red


def risk_label(score: int) -> str:
    if score < 35:
        return "LOW"
    elif score < 65:
        return "MODERATE"
    else:
        return "HIGH"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚾ PitchGuard AI")
    st.markdown("---")
    st.markdown('<p class="sidebar-label">Gemini API Key</p>', unsafe_allow_html=True)
    api_key = st.text_input("", type="password", placeholder="AIza...", label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p class="sidebar-label">Analysis Settings</p>', unsafe_allow_html=True)
    max_frames = st.slider("Frames to Extract", min_value=6, max_value=20, value=10)

    st.markdown("---")
    st.markdown('<p class="sidebar-label">About</p>', unsafe_allow_html=True)
    st.markdown("""
<small style="color:#6b8f78; line-height:1.7">
PitchGuard uses <b style="color:#4cff8f">MediaPipe Pose</b> for skeleton 
estimation and <b style="color:#4cff8f">Gemini 1.5 Flash</b> for 
biomechanical analysis. Upload a pitching video to receive a 
frame-by-frame injury risk report.
</small>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div>
<span class="tag">MediaPipe Pose</span>
<span class="tag">Gemini 1.5</span>
<span class="tag">OpenCV</span>
<span class="tag">Streamlit</span>
</div>
""", unsafe_allow_html=True)


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">PITCH<span class="hero-accent">GUARD</span> AI</div>
  <div class="hero-sub">Baseball Pitcher · Biomechanics Coach · Injury Risk Analyser</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload Pitching Video",
    type=["mp4", "mov", "avi", "mkv"],
    help="Upload a short clip (10–30 s) of the pitcher's full delivery for best results.",
)

if uploaded:
    st.video(uploaded)
    st.markdown("---")

    if not api_key:
        st.warning("⚠️  Please enter your Gemini API key in the sidebar to run the analysis.")
    else:
        if st.button("🔍  Analyse Pitch — Detect Injury Risk"):

            # ── Save video to temp file ───────────────────────────────────────
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                # STEP 1 — Extract frames
                with st.spinner("Extracting frames from video…"):
                    frames = extract_frames(tmp_path, max_frames=max_frames)

                if not frames:
                    st.error("Could not extract any frames from this video. Please try a different file.")
                    st.stop()

                st.success(f"✅  Extracted {len(frames)} frames")

                # STEP 2 — Pose estimation
                with st.spinner("Running MediaPipe skeleton pose estimation…"):
                    annotated_frames  = []
                    landmark_summaries = []
                    for i, frame in enumerate(frames):
                        ann, lm = draw_skeleton(frame)
                        annotated_frames.append(ann)
                        landmark_summaries.append({"frame": i + 1, "landmarks": lm})

                # STEP 3 — Display skeleton frames
                st.markdown('<div class="section-heading">SKELETON POSE ESTIMATION</div>', unsafe_allow_html=True)
                cols = st.columns(min(4, len(annotated_frames)))
                for i, ann in enumerate(annotated_frames):
                    with cols[i % len(cols)]:
                        st.image(ann, use_container_width=True)
                        st.markdown(f'<div class="frame-caption">Frame {i+1}</div>', unsafe_allow_html=True)

                st.markdown("---")

                # STEP 4 — Send to Gemini
                with st.spinner("Consulting PitchGuard AI coach (Gemini 1.5 Flash)…"):
                    frames_b64 = [frame_to_b64(f) for f in annotated_frames]
                    analysis   = call_gemini(api_key, frames_b64, landmark_summaries)

                # STEP 5 — Parse risk index
                risk_score = parse_risk_index(analysis)
                colour      = risk_colour(risk_score)
                label       = risk_label(risk_score)

                # STEP 6 — Injury Risk Index display
                st.markdown('<div class="section-heading">INJURY RISK INDEX</div>', unsafe_allow_html=True)

                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.markdown(f"""
<div class="risk-card" style="text-align:center; border-color:{colour}40;">
  <div class="risk-label">Overall Injury Risk Score</div>
  <div class="risk-value" style="font-size:5rem; color:{colour};">{risk_score}</div>
  <div style="font-family:'Bebas Neue',cursive; font-size:1.4rem; letter-spacing:0.2em; color:{colour}; margin-top:0.25rem;">
    {label} RISK
  </div>
  <div class="risk-bar-bg" style="margin-top:1rem;">
    <div class="risk-bar-fill" style="width:{risk_score}%; background:{colour};"></div>
  </div>
  <div style="display:flex; justify-content:space-between; margin-top:0.3rem; font-size:0.7rem; color:#6b8f78;">
    <span>0 — Safe</span><span>50 — Moderate</span><span>100 — Critical</span>
  </div>
</div>
""", unsafe_allow_html=True)

                # Risk legend
                lc1, lc2, lc3 = st.columns(3)
                lc1.markdown("""<div class="risk-bar-wrap">
  <div class="risk-label">Low Risk Zone</div>
  <div class="risk-value risk-low">0 – 34</div>
  <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:34%; background:#4cff8f;"></div></div>
</div>""", unsafe_allow_html=True)
                lc2.markdown("""<div class="risk-bar-wrap">
  <div class="risk-label">Moderate Risk Zone</div>
  <div class="risk-value risk-med">35 – 64</div>
  <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:64%; background:#ffd74c;"></div></div>
</div>""", unsafe_allow_html=True)
                lc3.markdown("""<div class="risk-bar-wrap">
  <div class="risk-label">High Risk Zone</div>
  <div class="risk-value risk-high">65 – 100</div>
  <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:100%; background:#ff5c5c;"></div></div>
</div>""", unsafe_allow_html=True)

                st.markdown("---")

                # STEP 7 — AI Coach Analysis
                st.markdown('<div class="section-heading">AI COACH ANALYSIS</div>', unsafe_allow_html=True)
                # Remove the RISK_INDEX line from displayed text
                clean_analysis = "\n".join(
                    l for l in analysis.splitlines() if "RISK_INDEX:" not in l
                )
                st.markdown(f'<div class="coach-box">{clean_analysis}</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("""
<div style="text-align:center; color:#3a6650; font-size:0.75rem; letter-spacing:0.15em; text-transform:uppercase; padding: 1rem 0;">
  PitchGuard AI · Not a substitute for professional medical advice · For coaching use only
</div>
""", unsafe_allow_html=True)

            finally:
                os.unlink(tmp_path)

else:
    # Empty state illustration
    st.markdown("""
<div style="text-align:center; padding:4rem 0; color:#3a6650;">
  <div style="font-size:5rem; margin-bottom:1rem;">⚾</div>
  <div style="font-family:'Bebas Neue',cursive; font-size:2rem; letter-spacing:0.1em; color:#4a7a60;">
    UPLOAD A PITCHING VIDEO TO BEGIN
  </div>
  <div style="font-size:0.85rem; margin-top:0.5rem; letter-spacing:0.1em; text-transform:uppercase;">
    Supports MP4 · MOV · AVI · MKV
  </div>
</div>
""", unsafe_allow_html=True)

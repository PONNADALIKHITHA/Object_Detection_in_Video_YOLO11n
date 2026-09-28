import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import streamlit as st
from ultralytics import YOLO

from object_detection_video import MODEL_NAME, process_video


PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_DIR / "input_videos"

st.set_page_config(
    page_title="YOLO11n Video Lab",
    page_icon="🎯",
    layout="wide",
)
st.markdown(
    """
    <style>
    :root { --ink: #172b2b; --muted: #627473; --line: #dce6e2; --accent: #087f68; }
    .block-container { max-width: 1180px; padding-top: 2.4rem; }
    h1, h2, h3 { color: var(--ink); }
    .eyebrow { color: var(--accent); font-size: .78rem; font-weight: 700;
               letter-spacing: .12em; text-transform: uppercase; }
    .lede { color: var(--muted); font-size: 1.05rem; }
    [data-testid="stMetric"] { background: #f2f7f4; border-left: 3px solid var(--accent);
                                padding: .85rem 1rem; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Computer vision · YOLO11n</div>', unsafe_allow_html=True)
st.title("Video detection lab")
st.markdown(
    '<p class="lede">Run object detection on a sample clip or your own video, then preview and download the annotated result.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Detections below this confidence are filtered out.",
    )
    st.caption("Lower values show more detections; higher values are more selective.")

samples = sorted(SAMPLE_DIR.glob("*.mp4"))
source_mode = st.radio("Video source", ["Sample video", "Upload a video"], horizontal=True)
uploaded_video = None
input_path = None
input_label = None

if source_mode == "Sample video":
    if samples:
        selected_sample = st.selectbox("Choose a sample", samples, format_func=lambda path: path.name)
        input_path = selected_sample
        input_label = selected_sample.name
    else:
        st.warning("No sample videos are included. Upload a video to continue.")
else:
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="The video is processed temporarily and is not saved to the server.",
    )
    if uploaded_video is not None:
        input_label = uploaded_video.name

if input_label:
    st.caption(f"Selected video: {input_label}")

run_detection = st.button(
    "Run detection",
    type="primary",
    disabled=(input_path is None and uploaded_video is None),
)

if run_detection:
    progress = st.progress(0, text="Loading YOLO11n model...")
    status = st.empty()
    try:
        @st.cache_resource
        def load_model():
            return YOLO(MODEL_NAME)

        model = load_model()
        with tempfile.TemporaryDirectory(prefix="yolo11n-") as temp_dir:
            temp_dir_path = Path(temp_dir)
            if uploaded_video is not None:
                input_path = temp_dir_path / Path(uploaded_video.name).name
                input_path.write_bytes(uploaded_video.getvalue())
            output_path = temp_dir_path / "annotated_video.mp4"

            def update_progress(frame, total_frames):
                if total_frames > 0:
                    progress.progress(
                        min(frame / total_frames, 1.0),
                        text=f"Processing frame {frame:,} of {total_frames:,}...",
                    )
                else:
                    status.write(f"Processed {frame:,} frames...")

            result = process_video(
                model,
                input_path,
                output_path,
                conf=confidence,
                progress_callback=update_progress,
            )
            browser_video_path = temp_dir_path / "annotated_video_h264.mp4"
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-y",
                    "-i",
                    str(output_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "28",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-threads",
                    "2",
                    str(browser_video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            video_bytes = browser_video_path.read_bytes()

        progress.progress(1.0, text="Detection complete")
        st.session_state["detection_result"] = result
        st.session_state["annotated_video"] = video_bytes
        st.session_state["result_source"] = input_label
        st.session_state["result_confidence"] = confidence
    except Exception as error:
        progress.empty()
        st.error(f"Could not process this video: {error}")

result = st.session_state.get("detection_result")
if result and st.session_state.get("result_source") == input_label and st.session_state.get("result_confidence") == confidence:
    st.divider()
    st.subheader("Detection result")
    metrics = st.columns(3)
    metrics[0].metric("Frames processed", f"{result['frames_processed']:,}")
    metrics[1].metric("Objects detected", f"{result['total_detections']:,}")
    metrics[2].metric("Classes found", len([name for name in result["classes_detected"].split(", ") if name]))

    class_names = result["classes_detected"] or "No objects passed the confidence threshold."
    st.caption(f"Detected classes: {class_names}")
    st.video(st.session_state["annotated_video"])
    download_name = f"{Path(input_label).stem}_detected.mp4"
    st.download_button(
        "Download annotated video",
        data=st.session_state["annotated_video"],
        file_name=download_name,
        mime="video/mp4",
        type="primary",
    )

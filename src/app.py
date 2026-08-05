from collections import Counter
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
from ultralytics import YOLO

try:
    from src.media_processing import (
        CLASS_NAMES,
        MAX_VIDEO_DURATION_SECONDS,
        VideoProcessingError,
        get_image_gps,
        get_media_kind,
        get_video_gps,
        infer_frame,
        process_video,
    )
except ModuleNotFoundError as exc:
    if exc.name != "src":
        raise
    from media_processing import (
        CLASS_NAMES,
        MAX_VIDEO_DURATION_SECONDS,
        VideoProcessingError,
        get_image_gps,
        get_media_kind,
        get_video_gps,
        infer_frame,
        process_video,
    )


MAX_UPLOAD_SIZE_MB = 100


st.set_page_config(page_title="AURA AI", page_icon="🛣️", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[1] / "models" / "best.pt"
    try:
        return YOLO(str(model_path)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def describe_counts(counts: Counter[int] | dict[int, int]) -> str:
    descriptions = [
        f"{count} **{CLASS_NAMES.get(class_id, 'Unknown')}** detection(s)"
        for class_id, count in sorted(counts.items())
    ]
    return ", and ".join(descriptions)


def render_location(
    gps_coordinates: tuple[float, float] | None,
    fallback_latitude: float,
    fallback_longitude: float,
    media_label: str,
) -> None:
    st.markdown("### 🗺️ Geospatial Hazard Mapping")

    if gps_coordinates is not None:
        latitude, longitude = gps_coordinates
        st.success(
            f"📍 Embedded GPS extracted from the {media_label}: "
            f"Lat {latitude:.4f}, Lon {longitude:.4f}"
        )
    else:
        latitude, longitude = fallback_latitude, fallback_longitude
        st.warning(
            "⚠️ No supported GPS metadata was found. Using the configured "
            f"fallback: Lat {latitude:.4f}, Lon {longitude:.4f}"
        )

    map_data = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
    st.map(map_data, zoom=14, width="stretch")


def render_image_analysis(
    uploaded_file,
    model,
    confidence_threshold: float,
    fallback_latitude: float,
    fallback_longitude: float,
) -> None:
    try:
        with Image.open(uploaded_file) as source_image:
            gps_coordinates = get_image_gps(source_image)
            display_image = ImageOps.exif_transpose(source_image).convert("RGB")
        frame_bgr = cv2.cvtColor(np.asarray(display_image), cv2.COLOR_RGB2BGR)
    except (OSError, UnidentifiedImageError, ValueError):
        st.error("❌ The uploaded image could not be read.")
        return

    try:
        with st.spinner("Neural network processing image..."):
            annotated_bgr, class_counts = infer_frame(
                model,
                frame_bgr,
                confidence_threshold,
            )
    except Exception as exc:
        st.error(f"❌ Image processing failed: {exc}")
        return

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Processed Road Image", width="stretch")

    number_of_detections = sum(class_counts.values())
    if number_of_detections == 0:
        st.success("✅ No detections at the current confidence threshold.")
        return

    st.error(f"🚨 Detected {number_of_detections} road-hazard object(s).")
    st.info(f"🗣️ **AI Analysis Report:** Identified {describe_counts(class_counts)}.")
    render_location(
        gps_coordinates,
        fallback_latitude,
        fallback_longitude,
        "image",
    )


def render_video_result(
    result: dict,
    fallback_latitude: float,
    fallback_longitude: float,
) -> None:
    summary = result["summary"]
    output_bytes = result["output_bytes"]

    st.video(output_bytes, format="video/mp4")
    st.download_button(
        "Download Annotated Video",
        data=output_bytes,
        file_name="aura_annotated.mp4",
        mime="video/mp4",
        on_click="ignore",
        width="stretch",
    )
    st.caption("The annotated H.264 export is silent; the original audio is omitted.")

    metric_columns = st.columns(3)
    metric_columns[0].metric("Frames Analyzed", f"{summary.frames_processed:,}")
    metric_columns[1].metric(
        "Frames With Detections",
        f"{summary.frames_with_detections:,}",
    )
    metric_columns[2].metric(
        "Detection Events",
        f"{summary.total_detection_events:,}",
    )

    if summary.total_detection_events == 0:
        st.success("✅ No detections at the current confidence threshold.")
        if result["gps_coordinates"] is not None:
            latitude, longitude = result["gps_coordinates"]
            st.caption(
                "Recording GPS was found, but no hazard pin was created: "
                f"{latitude:.4f}, {longitude:.4f}."
            )
        return

    st.error(
        "🚨 Detections appeared in "
        f"{summary.frames_with_detections:,} analyzed frame(s)."
    )
    st.info(
        "🗣️ **Frame-Level Analysis Report:** Identified "
        f"{describe_counts(summary.detection_events_by_class)}."
    )
    st.caption(
        "Detection events are bounding boxes counted per frame; the same physical "
        "road defect may appear in multiple frames."
    )
    render_location(
        result["gps_coordinates"],
        fallback_latitude,
        fallback_longitude,
        "video",
    )


def render_video_analysis(
    uploaded_file,
    model,
    confidence_threshold: float,
    fallback_latitude: float,
    fallback_longitude: float,
) -> None:
    uploaded_bytes = uploaded_file.getvalue()
    if len(uploaded_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        st.error(f"❌ Upload a video no larger than {MAX_UPLOAD_SIZE_MB} MB.")
        return
    if get_media_kind(uploaded_file.name) != "video":
        st.error("❌ Unsupported video type. Use MP4, MOV, or M4V.")
        return

    preview_format = getattr(uploaded_file, "type", None) or "video/mp4"
    st.video(uploaded_bytes, format=preview_format)
    st.caption(
        f"Recorded-video demo: maximum {int(MAX_VIDEO_DURATION_SECONDS)} seconds. "
        "Select Analyze Video to begin frame-by-frame inference."
    )

    if not st.button("Analyze Video", type="primary", width="stretch"):
        return

    progress_bar = st.progress(0.0, text="Preparing video...")
    preview = st.empty()
    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        with tempfile.TemporaryDirectory(prefix="aura-video-") as temp_directory:
            temp_dir = Path(temp_directory)
            input_path = temp_dir / f"input{suffix}"
            output_path = temp_dir / "aura_annotated.mp4"
            input_path.write_bytes(uploaded_bytes)
            gps_coordinates = get_video_gps(input_path)

            def update_progress(
                completed_frames: int,
                total_frames: int | None,
                annotated_bgr: np.ndarray,
            ) -> None:
                if total_frames:
                    fraction = min(completed_frames / total_frames, 1.0)
                    progress_text = (
                        f"Analyzing frame {completed_frames:,} of {total_frames:,}..."
                    )
                else:
                    fraction = 0.0
                    progress_text = f"Analyzing frame {completed_frames:,}..."
                progress_bar.progress(fraction, text=progress_text)
                preview.image(
                    cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                    caption="Processing Preview",
                    width="stretch",
                )

            summary = process_video(
                input_path,
                output_path,
                model,
                confidence_threshold,
                progress_callback=update_progress,
            )
            output_bytes = output_path.read_bytes()

        result = {
            "output_bytes": output_bytes,
            "summary": summary,
            "gps_coordinates": gps_coordinates,
        }
        progress_bar.progress(
            1.0,
            text=f"Analyzed {summary.frames_processed:,} frames.",
        )
    except VideoProcessingError as exc:
        progress_bar.empty()
        preview.empty()
        st.error(f"❌ {exc}")
        return
    except OSError as exc:
        progress_bar.empty()
        preview.empty()
        st.error(f"❌ Video file handling failed: {exc}")
        return

    render_video_result(
        result,
        fallback_latitude,
        fallback_longitude,
    )


model, model_error = load_model()
model_loaded = model is not None

st.title("🛣️ AURA: Smart City Infrastructure AI")
st.markdown("**Team 14D** | Automated Urban Road Assessment System")
st.divider()

st.sidebar.title("⚙️ System Controls")
confidence_threshold = st.sidebar.slider(
    "AI Confidence Threshold",
    0.1,
    1.0,
    0.25,
    help="Lower this to catch faint cracks. Raise it to ignore shadows.",
)

st.sidebar.markdown("### 📍 Location Metadata Fallback")
st.sidebar.markdown(
    "*Used when uploaded image/video metadata does not contain supported GPS data.*"
)
fallback_latitude = st.sidebar.number_input(
    "Fallback Latitude",
    value=38.8951,
    format="%.4f",
)
fallback_longitude = st.sidebar.number_input(
    "Fallback Longitude",
    value=-77.0364,
    format="%.4f",
)

input_column, output_column = st.columns([1, 2])

with input_column:
    st.markdown("### 📸 1. Road Footage")
    media_mode = st.radio(
        "Choose media type",
        ("Image", "Video"),
        horizontal=True,
    )

    if media_mode == "Image":
        st.markdown("Upload a road image to scan for structural degradation.")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            max_upload_size=MAX_UPLOAD_SIZE_MB,
            key="image-upload",
        )
    else:
        st.markdown("Upload or record a short road video for frame-by-frame analysis.")
        uploaded_file = st.file_uploader(
            "Upload or record a video",
            type="video",
            max_upload_size=MAX_UPLOAD_SIZE_MB,
            key="video-upload",
            help=(
                "Your phone's native picker may offer Record Video. The exact options "
                "depend on the browser and operating system."
            ),
        )
        st.caption("Supports MP4, MOV, and M4V; up to 30 seconds and 100 MB.")

with output_column:
    st.markdown("### 🖥️ 2. Inference Output")

    if uploaded_file is None:
        st.info(f"Awaiting a road {media_mode.lower()} upload...")
    elif not model_loaded:
        st.error(f"❌ AI model failed to load: {model_error}")
    elif media_mode == "Image":
        render_image_analysis(
            uploaded_file,
            model,
            confidence_threshold,
            fallback_latitude,
            fallback_longitude,
        )
    else:
        render_video_analysis(
            uploaded_file,
            model,
            confidence_threshold,
            fallback_latitude,
            fallback_longitude,
        )

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


st.set_page_config(
    page_title="AURA AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": (
            "AURA is a road-hazard detection demo built by Team 14D for the "
            "AI4ALL Ignite Program."
        )
    },
)

# Streamlit keeps its native sidebar toggle; this marks the collapsed control area
# as Settings without relying on brittle internal CSS selectors.
st.logo(
    ":material/settings:",
    size="medium",
    icon_image=":material/settings:",
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
        f"{count} **{CLASS_NAMES.get(class_id, 'Unknown')}**"
        for class_id, count in sorted(counts.items())
    ]
    if len(descriptions) < 2:
        return "".join(descriptions)
    if len(descriptions) == 2:
        return " and ".join(descriptions)
    return f"{', '.join(descriptions[:-1])}, and {descriptions[-1]}"


def render_stat_cards(statistics: list[str], card_width: int) -> None:
    statistic_row = st.container(horizontal=True, gap="small")
    for statistic in statistics:
        with statistic_row.container(border=True, width=card_width):
            st.markdown(f"**{statistic}**")


def render_location(
    gps_coordinates: tuple[float, float] | None,
    fallback_latitude: float,
    fallback_longitude: float,
    media_label: str,
) -> None:
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


def analyze_image(
    uploaded_file,
    model,
    confidence_threshold: float,
) -> dict | None:
    try:
        with Image.open(uploaded_file) as source_image:
            gps_coordinates = get_image_gps(source_image)
            display_image = ImageOps.exif_transpose(source_image).convert("RGB")
        frame_bgr = cv2.cvtColor(np.asarray(display_image), cv2.COLOR_RGB2BGR)
    except (OSError, UnidentifiedImageError, ValueError):
        st.error("❌ The uploaded image could not be read.")
        return None

    try:
        with st.spinner("Neural network processing image..."):
            annotated_bgr, class_counts = infer_frame(
                model,
                frame_bgr,
                confidence_threshold,
            )
    except Exception as exc:
        st.error(f"❌ Image processing failed: {exc}")
        return None

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return {
        "annotated_rgb": annotated_rgb,
        "class_counts": class_counts,
        "gps_coordinates": gps_coordinates,
    }


def render_image_result(result: dict) -> None:
    annotated_rgb = result["annotated_rgb"]
    class_counts = result["class_counts"]

    st.image(annotated_rgb, width="stretch")

    number_of_detections = sum(class_counts.values())
    number_of_damage_types = len(class_counts)
    detection_label = "detection" if number_of_detections == 1 else "detections"
    damage_type_label = (
        "damage type" if number_of_damage_types == 1 else "damage types"
    )

    render_stat_cards(
        [
            f"{number_of_detections:,} {detection_label}",
            f"{number_of_damage_types:,} {damage_type_label}",
        ],
        card_width=220,
    )

    if number_of_detections == 0:
        with st.container(border=True):
            st.markdown("✅ No road damage found at the current confidence threshold.")
        return

    with st.container(border=True):
        st.markdown(
            f"🗣️ **AI analysis:** Identified {describe_counts(class_counts)}."
        )


def render_video_result(
    result: dict,
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
    st.caption("Silent H.264 export; original audio omitted.")

    number_of_damage_types = len(summary.detection_events_by_class)
    frame_label = "frame" if summary.frames_processed == 1 else "frames"
    detected_frame_label = (
        "frame" if summary.frames_with_detections == 1 else "frames"
    )
    event_label = (
        "detection event"
        if summary.total_detection_events == 1
        else "detection events"
    )
    damage_type_label = (
        "damage type" if number_of_damage_types == 1 else "damage types"
    )

    render_stat_cards(
        [
            f"{summary.frames_processed:,} {frame_label} analyzed",
            (
                f"{summary.frames_with_detections:,} {detected_frame_label} "
                "with detections"
            ),
            f"{summary.total_detection_events:,} {event_label}",
            f"{number_of_damage_types:,} {damage_type_label}",
        ],
        card_width=155,
    )

    if summary.total_detection_events == 0:
        with st.container(border=True):
            st.markdown("✅ No road damage found at the current confidence threshold.")
        return

    with st.container(border=True):
        st.markdown(
            "🗣️ **Frame-level analysis:** Identified "
            f"{describe_counts(summary.detection_events_by_class)}."
        )
        st.caption(
            "Detection events are bounding boxes counted per frame; the same physical "
            "road defect may appear in multiple frames."
        )


def analyze_video(
    uploaded_file,
    model,
    confidence_threshold: float,
) -> dict | None:
    uploaded_bytes = uploaded_file.getvalue()
    if len(uploaded_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        st.error(f"❌ Upload a video no larger than {MAX_UPLOAD_SIZE_MB} MB.")
        return None
    if get_media_kind(uploaded_file.name) != "video":
        st.error("❌ Unsupported video type. Use MP4, MOV, or M4V.")
        return None

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
        return None
    except OSError as exc:
        progress_bar.empty()
        preview.empty()
        st.error(f"❌ Video file handling failed: {exc}")
        return None

    return result


model, model_error = load_model()
model_loaded = model is not None

with st.sidebar:
    st.title("⚙️ Settings")
    st.caption(
        "This panel starts closed. Reopen it from the top-left control beside "
        "the cog. On phones it opens over the app."
    )
    st.divider()

    st.subheader("Detection")
    confidence_threshold = st.slider(
        "Confidence threshold",
        0.1,
        1.0,
        0.25,
        help="Lower values catch fainter damage; higher values reduce false positives.",
    )
    st.caption(
        f"Showing predictions at or above **{confidence_threshold:.0%}** confidence."
    )

    with st.expander("Location fallback", icon=":material/location_on:"):
        st.caption(
            "Used only when the uploaded image or video has no supported GPS metadata."
        )
        fallback_latitude = st.number_input(
            "Fallback latitude",
            value=38.8951,
            format="%.4f",
        )
        fallback_longitude = st.number_input(
            "Fallback longitude",
            value=-77.0364,
            format="%.4f",
        )

    st.divider()
    if model_loaded:
        st.success("YOLOv8s model ready", icon="✅")
    else:
        st.error("Model unavailable", icon="❌")
    st.caption("Recorded-video limit: 30 seconds · 100 MB")

page_shell = st.container(
    horizontal=True,
    horizontal_alignment="center",
)
page = page_shell.container(width=720)

with page:
    with st.container(border=True):
        st.title("🛣️ AURA")
        st.markdown("**Automated Urban Road Assessment** · Team 14D")
        st.caption(
            "Upload road footage, detect pavement damage, and map available GPS "
            "metadata."
        )
        with st.container(horizontal=True, gap="small"):
            st.badge("YOLOv8s", icon=":material/neurology:", color="blue")
            st.badge(
                "Images + recorded video",
                icon=":material/video_camera_back:",
                color="violet",
            )
            st.badge("GPS metadata", icon=":material/location_on:", color="green")

    with st.container(border=True):
        st.subheader("1. Upload image or video")
        media_mode = st.segmented_control(
            "Media type",
            ("Image", "Video"),
            default="Image",
            required=True,
            format_func=lambda mode: "📷 Image" if mode == "Image" else "🎥 Video",
            width="stretch",
            key="media-mode",
        )

        if media_mode == "Image":
            st.caption("Upload a road image for a single-frame inspection.")
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png"],
                max_upload_size=MAX_UPLOAD_SIZE_MB,
                key="image-upload",
            )
            st.caption("JPG, JPEG, or PNG · maximum 100 MB")
        else:
            st.caption("Upload or record a short clip for frame-by-frame inspection.")
            uploaded_file = st.file_uploader(
                "Upload or record a video",
                type="video",
                max_upload_size=MAX_UPLOAD_SIZE_MB,
                key="video-upload",
                help=(
                    "Your phone's native picker may offer Record Video. The exact "
                    "options depend on the browser and operating system."
                ),
            )
            st.caption("MP4, MOV, or M4V · maximum 30 seconds / 100 MB")

    analyze_clicked = False
    with st.container(border=True):
        st.subheader("2. Review upload")

        if uploaded_file is None:
            st.info(f"Upload a road {media_mode.lower()} to preview it.", icon="📤")
        else:
            uploaded_bytes = uploaded_file.getvalue()
            if media_mode == "Image":
                st.image(
                    uploaded_bytes,
                    caption="Original road image",
                    width="stretch",
                )
            else:
                preview_format = getattr(uploaded_file, "type", None) or "video/mp4"
                st.video(uploaded_bytes, format=preview_format)
                st.caption(
                    "Recorded-video demo: maximum "
                    f"{int(MAX_VIDEO_DURATION_SECONDS)} seconds."
                )

            if not model_loaded:
                st.error(f"AI model failed to load: {model_error}", icon="❌")

            analyze_clicked = st.button(
                f"Analyze {media_mode}",
                type="primary",
                width="stretch",
                disabled=not model_loaded,
            )

    analysis_result = None
    result_section_number = 3 if media_mode == "Image" else 4
    location_section_number = result_section_number + 1

    if analyze_clicked and media_mode == "Video":
        with st.container(border=True):
            st.subheader("3. Processing preview")
            st.caption(
                "Bounding boxes appear here while AURA analyzes every video frame."
            )
            analysis_result = analyze_video(
                uploaded_file,
                model,
                confidence_threshold,
            )

    if analyze_clicked and media_mode == "Image":
        with st.container(border=True):
            st.subheader(
                f"{result_section_number}. Annotated Image"
            )
            analysis_result = analyze_image(
                uploaded_file,
                model,
                confidence_threshold,
            )
            if analysis_result is not None:
                render_image_result(analysis_result)
    elif analysis_result is not None:
        with st.container(border=True):
            st.subheader(
                f"{result_section_number}. Annotated video + statistics"
            )
            render_video_result(analysis_result)

    if analysis_result is not None:
        with st.container(border=True):
            st.subheader(f"{location_section_number}. Location & map")
            render_location(
                analysis_result["gps_coordinates"],
                fallback_latitude,
                fallback_longitude,
                media_mode.lower(),
            )

"""Media decoding, inference, metadata, and video encoding helpers for AURA."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Literal

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import ExifTags, Image
from pymediainfo import MediaInfo


CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}

VIDEO_EXTENSIONS = frozenset({"mp4", "mov", "m4v"})
MAX_VIDEO_DURATION_SECONDS = 30.0

_IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png"})
_DEFAULT_VIDEO_FPS = 30.0
_GPS_INFO_TAG = 34853
_LEGACY_VIDEO_GPS_KEYS = frozenset({"gpscoordinates", "xyz"})
_ISO6709_PATTERN = re.compile(
    r"(?P<latitude>[+-]\d{1,2}(?:\.\d+)?)"
    r"(?P<longitude>[+-]\d{1,3}(?:\.\d+)?)"
    r"(?:[+-]\d+(?:\.\d+)?)?/?"
)

ProgressCallback = Callable[[int, int, np.ndarray], None]


class VideoProcessingError(RuntimeError):
    """Raised when an uploaded video cannot be safely processed."""


@dataclass(frozen=True)
class VideoSummary:
    """Frame-level video inference statistics.

    Detection events are bounding boxes observed across frames. They are not
    counts of unique physical hazards because the same object can appear in
    multiple frames.
    """

    frames_processed: int
    frames_with_detections: int
    detection_events_by_class: dict[int, int]
    frames_by_class: dict[int, int]
    fps: float
    width: int
    height: int
    duration_seconds: float

    @property
    def total_detection_events(self) -> int:
        """Return the total number of frame-level bounding-box detections."""

        return sum(self.detection_events_by_class.values())


def get_media_kind(filename: str | Path) -> Literal["image", "video"] | None:
    """Classify a supported filename by its case-insensitive extension."""

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension in _IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    return None


def _gps_coordinate_to_decimal(
    coordinate: Sequence[object], reference: object
) -> float:
    if len(coordinate) < 3:
        raise ValueError("GPS coordinate must contain degrees, minutes, and seconds")

    ref = (
        reference.decode("ascii", errors="ignore")
        if isinstance(reference, bytes)
        else str(reference)
    )
    decimal = (
        float(coordinate[0])
        + float(coordinate[1]) / 60.0
        + float(coordinate[2]) / 3600.0
    )
    if ref.strip().upper() in {"S", "W"}:
        decimal = -decimal
    return decimal


def get_image_gps(image: Image.Image) -> tuple[float, float] | None:
    """Extract decimal latitude and longitude from a Pillow image's EXIF."""

    try:
        exif = image.getexif()
        if not exif:
            return None

        gps_info = None
        try:
            gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except (AttributeError, KeyError, TypeError, ValueError):
            raw_gps_info = exif.get(_GPS_INFO_TAG)
            if isinstance(raw_gps_info, Mapping):
                gps_info = raw_gps_info

        if not isinstance(gps_info, Mapping):
            return None

        latitude = _gps_coordinate_to_decimal(gps_info[2], gps_info[1])
        longitude = _gps_coordinate_to_decimal(gps_info[4], gps_info[3])
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            return None
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            return None
        return latitude, longitude
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def parse_iso6709(value: object) -> tuple[float, float] | None:
    """Parse the latitude and longitude from a QuickTime ISO 6709 value."""

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if not isinstance(value, str):
        return None

    match = _ISO6709_PATTERN.search(value.strip().replace("\x00", ""))
    if match is None:
        return None

    try:
        latitude = float(match.group("latitude"))
        longitude = float(match.group("longitude"))
    except (TypeError, ValueError):
        return None

    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None
    return latitude, longitude


def _iter_location_values(value: object, location_context: bool = False) -> Iterator[object]:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            is_location = (
                location_context
                or "location" in normalized_key
                or "iso6709" in normalized_key
                or normalized_key in _LEGACY_VIDEO_GPS_KEYS
            )
            yield from _iter_location_values(nested_value, is_location)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested_value in value:
            yield from _iter_location_values(nested_value, location_context)
        return

    if location_context:
        yield value


def get_video_gps(path: str | Path) -> tuple[float, float] | None:
    """Extract one embedded recording location from video container metadata."""

    try:
        media_info = MediaInfo.parse(str(path))
        for track in media_info.tracks:
            for value in _iter_location_values(track.to_data()):
                coordinates = parse_iso6709(value)
                if coordinates is not None:
                    return coordinates
    except Exception:
        # Location metadata is optional and should never block inference.
        return None
    return None


def _validate_confidence(confidence: float) -> float:
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise VideoProcessingError("Confidence must be a number between 0 and 1.") from exc
    if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise VideoProcessingError("Confidence must be a number between 0 and 1.")
    return confidence_value


def infer_frame(
    model: object, frame_bgr: np.ndarray, confidence: float
) -> tuple[np.ndarray, Counter[int]]:
    """Run inference on one BGR frame and return its BGR annotation and counts."""

    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
    ):
        raise VideoProcessingError("Inference requires a three-channel BGR frame.")

    confidence_value = _validate_confidence(confidence)
    try:
        results = model.predict(source=frame_bgr, conf=confidence_value, verbose=False)
        if not results:
            raise VideoProcessingError("The model returned no inference result for a frame.")

        result = results[0]
        annotated_bgr = result.plot()
        if (
            not isinstance(annotated_bgr, np.ndarray)
            or annotated_bgr.ndim != 3
            or annotated_bgr.shape[2] != 3
            or annotated_bgr.dtype != np.uint8
        ):
            raise VideoProcessingError("The model returned an invalid annotated frame.")

        boxes = getattr(result, "boxes", None)
        classes = getattr(boxes, "cls", None) if boxes is not None else None
        if classes is None:
            return annotated_bgr, Counter()

        if hasattr(classes, "detach"):
            classes = classes.detach()
        if hasattr(classes, "cpu"):
            classes = classes.cpu()
        if hasattr(classes, "tolist"):
            classes = classes.tolist()

        flattened_classes = np.asarray(classes).reshape(-1).tolist()
        return annotated_bgr, Counter(int(class_id) for class_id in flattened_classes)
    except VideoProcessingError:
        raise
    except Exception as exc:
        raise VideoProcessingError(f"Model inference failed: {exc}") from exc


def _valid_video_fps(capture: cv2.VideoCapture) -> float:
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    except (TypeError, ValueError):
        return _DEFAULT_VIDEO_FPS
    if not math.isfinite(fps) or fps <= 0.0:
        return _DEFAULT_VIDEO_FPS
    return fps


def _reported_frame_count(capture: cv2.VideoCapture) -> int:
    try:
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(frame_count) or frame_count <= 0.0:
        return 0
    return int(frame_count)


def _pad_to_even_dimensions(frame_bgr: np.ndarray) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    bottom_padding = height % 2
    right_padding = width % 2
    if bottom_padding == 0 and right_padding == 0:
        return frame_bgr
    return cv2.copyMakeBorder(
        frame_bgr,
        0,
        bottom_padding,
        0,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    model: object,
    confidence: float,
    progress_callback: ProgressCallback | None = None,
    max_duration_seconds: float = MAX_VIDEO_DURATION_SECONDS,
) -> VideoSummary:
    """Incrementally annotate a video and atomically write an H.264 MP4.

    The callback receives ``(frames_processed, reported_total_frames,
    annotated_bgr)`` for the first and final frames and approximately five
    times per second in between. Every frame is still analyzed and encoded.
    """

    source_path = Path(input_path)
    destination_path = Path(output_path)
    if not source_path.is_file():
        raise VideoProcessingError("The uploaded video file does not exist.")
    if get_media_kind(source_path) != "video":
        raise VideoProcessingError("Unsupported video format. Use MP4, MOV, or M4V.")
    if destination_path.suffix.lower() != ".mp4":
        raise VideoProcessingError("The annotated output must use the MP4 extension.")
    if source_path.resolve() == destination_path.resolve():
        raise VideoProcessingError("Input and output video paths must be different.")

    try:
        duration_limit = float(max_duration_seconds)
    except (TypeError, ValueError) as exc:
        raise VideoProcessingError("Maximum video duration must be positive.") from exc
    if not math.isfinite(duration_limit) or duration_limit <= 0.0:
        raise VideoProcessingError("Maximum video duration must be positive.")
    confidence_value = _validate_confidence(confidence)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError("The uploaded video could not be opened or decoded.")

    fps = _valid_video_fps(capture)
    reported_total_frames = _reported_frame_count(capture)
    if reported_total_frames and reported_total_frames / fps > duration_limit + 1e-9:
        capture.release()
        raise VideoProcessingError(
            f"Video exceeds the {duration_limit:g}-second duration limit."
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination_path.stem}.",
        suffix=".mp4",
        dir=str(destination_path.parent),
    )
    os.close(temp_descriptor)
    temporary_output_path = Path(temp_name)

    encoder = None
    frames_processed = 0
    frames_with_detections = 0
    detection_events: Counter[int] = Counter()
    frames_by_class: Counter[int] = Counter()
    encoded_width = 0
    encoded_height = 0
    max_frames = max(1, math.floor(duration_limit * fps + 1e-9))
    progress_interval_frames = max(1, round(fps / 5.0))
    last_annotated_bgr: np.ndarray | None = None
    last_progress_frame = 0

    try:
        try:
            while True:
                decoded, frame_bgr = capture.read()
                if not decoded:
                    break
                if frames_processed >= max_frames:
                    raise VideoProcessingError(
                        f"Video exceeds the {duration_limit:g}-second duration limit."
                    )

                annotated_bgr, frame_counts = infer_frame(
                    model, frame_bgr, confidence_value
                )
                encoded_frame = _pad_to_even_dimensions(annotated_bgr)

                if encoder is None:
                    encoded_height, encoded_width = encoded_frame.shape[:2]
                    encoder = imageio_ffmpeg.write_frames(
                        str(temporary_output_path),
                        (encoded_width, encoded_height),
                        pix_fmt_in="bgr24",
                        pix_fmt_out="yuv420p",
                        fps=fps,
                        quality=7,
                        codec="libx264",
                        macro_block_size=1,
                        ffmpeg_log_level="error",
                        output_params=["-movflags", "+faststart"],
                    )
                    encoder.send(None)
                elif encoded_frame.shape[:2] != (encoded_height, encoded_width):
                    raise VideoProcessingError(
                        "Video frame dimensions changed during processing."
                    )

                encoder.send(np.ascontiguousarray(encoded_frame))
                frames_processed += 1
                last_annotated_bgr = annotated_bgr
                detection_events.update(frame_counts)
                if frame_counts:
                    frames_with_detections += 1
                    frames_by_class.update(frame_counts.keys())

                should_report_progress = (
                    frames_processed == 1
                    or (frames_processed - 1) % progress_interval_frames == 0
                )
                if progress_callback is not None and should_report_progress:
                    progress_callback(
                        frames_processed, reported_total_frames, annotated_bgr
                    )
                    last_progress_frame = frames_processed

            if (
                progress_callback is not None
                and last_annotated_bgr is not None
                and last_progress_frame != frames_processed
            ):
                progress_callback(
                    frames_processed, reported_total_frames, last_annotated_bgr
                )
        finally:
            capture.release()
            if encoder is not None:
                encoder.close()

        if frames_processed == 0:
            raise VideoProcessingError("The uploaded video contains no decodable frames.")
        if not temporary_output_path.is_file() or temporary_output_path.stat().st_size == 0:
            raise VideoProcessingError("FFmpeg did not produce an annotated video.")

        os.replace(temporary_output_path, destination_path)
        return VideoSummary(
            frames_processed=frames_processed,
            frames_with_detections=frames_with_detections,
            detection_events_by_class=dict(sorted(detection_events.items())),
            frames_by_class=dict(sorted(frames_by_class.items())),
            fps=fps,
            width=encoded_width,
            height=encoded_height,
            duration_seconds=frames_processed / fps,
        )
    except VideoProcessingError:
        raise
    except Exception as exc:
        raise VideoProcessingError(f"Video processing failed: {exc}") from exc
    finally:
        try:
            temporary_output_path.unlink(missing_ok=True)
        except OSError:
            pass

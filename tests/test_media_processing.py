from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, TiffImagePlugin

from src.media_processing import (
    CLASS_NAMES,
    MAX_VIDEO_DURATION_SECONDS,
    VIDEO_EXTENSIONS,
    VideoProcessingError,
    get_image_gps,
    get_media_kind,
    get_video_gps,
    infer_frame,
    parse_iso6709,
    process_video,
)


class _FakeBoxes:
    def __init__(self, classes: list[int]) -> None:
        self.cls = np.asarray(classes, dtype=np.float32)


class _FakeResult:
    def __init__(self, frame_bgr: np.ndarray, classes: list[int]) -> None:
        self._frame_bgr = frame_bgr
        self.boxes = _FakeBoxes(classes)

    def plot(self) -> np.ndarray:
        return self._frame_bgr.copy()


class _SequencedModel:
    def __init__(self, classes_by_frame: list[list[int]]) -> None:
        self._classes_by_frame = iter(classes_by_frame)
        self.predict_calls: list[tuple[np.ndarray, float, bool]] = []

    def predict(
        self, *, source: np.ndarray, conf: float, verbose: bool
    ) -> list[_FakeResult]:
        self.predict_calls.append((source.copy(), conf, verbose))
        return [_FakeResult(source, next(self._classes_by_frame))]


def _write_h264_video(
    path: Path, *, frame_count: int, fps: float, size: tuple[int, int] = (64, 48)
) -> None:
    """Create a small, deterministic H.264 fixture without a system ffmpeg."""
    width, height = size
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        ffmpeg_log_level="error",
    )
    writer.send(None)
    try:
        for index in range(frame_count):
            frame_rgb = np.empty((height, width, 3), dtype=np.uint8)
            frame_rgb[:, :, 0] = (index * 31) % 256
            frame_rgb[:, :, 1] = (index * 17) % 256
            frame_rgb[:, :, 2] = (index * 7) % 256
            writer.send(frame_rgb.tobytes())
    finally:
        writer.close()


class MediaDispatchTests(unittest.TestCase):
    def test_dispatches_supported_image_and_video_extensions(self) -> None:
        for filename in ("road.jpg", "road.JPEG", "road.png"):
            with self.subTest(filename=filename):
                self.assertEqual(get_media_kind(filename), "image")

        for extension in VIDEO_EXTENSIONS:
            with self.subTest(extension=extension):
                self.assertEqual(get_media_kind(f"clip.{extension.upper()}"), "video")

    def test_rejects_unknown_or_missing_extensions(self) -> None:
        for filename in ("clip.avi", "notes.txt", "no-extension", ""):
            with self.subTest(filename=filename):
                self.assertIsNone(get_media_kind(filename))

    def test_public_video_contract_has_demo_limit_and_expected_formats(self) -> None:
        self.assertEqual(VIDEO_EXTENSIONS, frozenset({"mp4", "mov", "m4v"}))
        self.assertEqual(MAX_VIDEO_DURATION_SECONDS, 30.0)
        self.assertEqual(set(CLASS_NAMES), {0, 1, 2, 3})


class LocationMetadataTests(unittest.TestCase):
    def test_parse_iso6709_handles_hemispheres_and_altitude(self) -> None:
        cases = {
            "+38.8951-077.0364+000.0/": (38.8951, -77.0364),
            "-33.8688+151.2093/": (-33.8688, 151.2093),
            "+00.0000-000.0000/": (0.0, -0.0),
        }
        for encoded, expected in cases.items():
            with self.subTest(encoded=encoded):
                actual = parse_iso6709(encoded)
                self.assertIsNotNone(actual)
                self.assertAlmostEqual(actual[0], expected[0], places=6)
                self.assertAlmostEqual(actual[1], expected[1], places=6)

    def test_parse_iso6709_rejects_invalid_or_out_of_range_values(self) -> None:
        for value in (None, "", "not-a-location", "+91.0-077.0/", "+38.0-181.0/"):
            with self.subTest(value=value):
                self.assertIsNone(parse_iso6709(value))

    def test_extracts_gps_from_real_jpeg_exif(self) -> None:
        rational = TiffImagePlugin.IFDRational
        exif = Image.Exif()
        exif[34853] = {
            1: "N",
            2: (rational(38), rational(53), rational(4212, 100)),
            3: "W",
            4: (rational(77), rational(2), rational(1104, 100)),
        }
        buffer = BytesIO()
        Image.new("RGB", (4, 4), "white").save(buffer, format="JPEG", exif=exif)
        buffer.seek(0)

        with Image.open(buffer) as image:
            gps = get_image_gps(image)

        self.assertIsNotNone(gps)
        self.assertAlmostEqual(gps[0], 38.8950333333, places=6)
        self.assertAlmostEqual(gps[1], -77.0364, places=6)

    def test_image_without_gps_returns_none(self) -> None:
        self.assertIsNone(get_image_gps(Image.new("RGB", (2, 2))))

    @patch("src.media_processing.MediaInfo.parse")
    def test_extracts_video_gps_from_mediainfo(self, parse: Mock) -> None:
        track = Mock()
        track.to_data.return_value = {
            "com.apple.quicktime.location.ISO6709": "+38.8951-077.0364+000.0/"
        }
        parse.return_value = SimpleNamespace(tracks=[track])

        self.assertEqual(get_video_gps(Path("phone.mov")), (38.8951, -77.0364))
        parse.assert_called_once_with("phone.mov")

    @patch("src.media_processing.MediaInfo.parse")
    def test_extracts_video_gps_from_legacy_xyz_metadata(self, parse: Mock) -> None:
        track = Mock()
        track.to_data.return_value = {"xyz": "+38.8951-077.0364+000.0/"}
        parse.return_value = SimpleNamespace(tracks=[track])

        self.assertEqual(get_video_gps("legacy.mov"), (38.8951, -77.0364))
        parse.assert_called_once_with("legacy.mov")

    @patch("src.media_processing.MediaInfo.parse", side_effect=RuntimeError("bad metadata"))
    def test_video_metadata_failure_is_nonfatal(self, parse: Mock) -> None:
        self.assertIsNone(get_video_gps("broken.mov"))
        parse.assert_called_once_with("broken.mov")


class FrameInferenceTests(unittest.TestCase):
    def test_inference_preserves_bgr_input_and_counts_classes(self) -> None:
        frame_bgr = np.array([[[3, 17, 241], [80, 20, 7]]], dtype=np.uint8)
        annotated_bgr = np.array([[[9, 8, 7], [6, 5, 4]]], dtype=np.uint8)
        result = Mock()
        result.boxes.cls = np.asarray([3, 3, 1], dtype=np.float32)
        result.plot.return_value = annotated_bgr
        model = Mock()
        model.predict.return_value = [result]

        annotated, counts = infer_frame(model, frame_bgr, confidence=0.42)

        kwargs = model.predict.call_args.kwargs
        self.assertIs(kwargs["source"], frame_bgr)
        self.assertEqual(kwargs["conf"], 0.42)
        self.assertFalse(kwargs["verbose"])
        np.testing.assert_array_equal(annotated, annotated_bgr)
        self.assertEqual(counts, Counter({3: 2, 1: 1}))


class VideoProcessingTests(unittest.TestCase):
    def test_processes_every_frame_aggregates_counts_and_writes_readable_h264(self) -> None:
        classes = [[3, 3], [], [3, 1], [1]]
        model = _SequencedModel(classes)
        progress: list[tuple[int, int, tuple[int, ...]]] = []

        with tempfile.TemporaryDirectory(prefix="aura-video-test-") as directory:
            input_path = Path(directory) / "input.mp4"
            output_path = Path(directory) / "output.mp4"
            _write_h264_video(input_path, frame_count=4, fps=4.0)

            summary = process_video(
                input_path,
                output_path,
                model,
                confidence=0.25,
                progress_callback=lambda current, total, frame: progress.append(
                    (current, total, frame.shape)
                ),
            )

            self.assertEqual(len(model.predict_calls), 4)
            self.assertEqual(summary.frames_processed, 4)
            self.assertEqual(summary.frames_with_detections, 3)
            self.assertEqual(summary.detection_events_by_class, {3: 3, 1: 2})
            self.assertEqual(summary.frames_by_class, {3: 2, 1: 2})
            self.assertEqual(summary.total_detection_events, 5)
            self.assertAlmostEqual(summary.fps, 4.0, places=2)
            self.assertEqual((summary.width, summary.height), (64, 48))
            self.assertAlmostEqual(summary.duration_seconds, 1.0, places=2)
            self.assertEqual([item[0] for item in progress], [1, 2, 3, 4])
            self.assertTrue(all(item[1] == 4 for item in progress))
            self.assertTrue(all(item[2] == (48, 64, 3) for item in progress))

            capture = cv2.VideoCapture(str(output_path))
            try:
                self.assertTrue(capture.isOpened())
                codec_number = int(capture.get(cv2.CAP_PROP_FOURCC))
                codec = "".join(
                    chr((codec_number >> (8 * index)) & 0xFF) for index in range(4)
                ).lower()
                decoded_frames = []
                while True:
                    readable, frame = capture.read()
                    if not readable:
                        break
                    decoded_frames.append(frame)
            finally:
                capture.release()

            self.assertIn(codec, {"avc1", "h264"})
            self.assertEqual(len(decoded_frames), 4)
            self.assertTrue(all(frame.shape == (48, 64, 3) for frame in decoded_frames))

    def test_progress_preview_is_throttled_without_skipping_inference(self) -> None:
        model = _SequencedModel([[] for _ in range(30)])
        progress_frames: list[int] = []

        with tempfile.TemporaryDirectory(prefix="aura-video-test-") as directory:
            input_path = Path(directory) / "input.mp4"
            output_path = Path(directory) / "output.mp4"
            _write_h264_video(input_path, frame_count=30, fps=30.0)

            summary = process_video(
                input_path,
                output_path,
                model,
                confidence=0.25,
                progress_callback=lambda current, _total, _frame: progress_frames.append(
                    current
                ),
            )

        self.assertEqual(summary.frames_processed, 30)
        self.assertEqual(len(model.predict_calls), 30)
        self.assertEqual(progress_frames, [1, 7, 13, 19, 25, 30])

    def test_corrupt_video_raises_domain_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aura-video-test-") as directory:
            input_path = Path(directory) / "corrupt.mp4"
            output_path = Path(directory) / "output.mp4"
            input_path.write_bytes(b"this is not a video")

            with self.assertRaises(VideoProcessingError):
                process_video(input_path, output_path, Mock(), confidence=0.25)

    def test_rejects_video_longer_than_thirty_seconds_before_inference(self) -> None:
        model = Mock()
        with tempfile.TemporaryDirectory(prefix="aura-video-test-") as directory:
            input_path = Path(directory) / "too-long.mp4"
            output_path = Path(directory) / "output.mp4"
            _write_h264_video(input_path, frame_count=31, fps=1.0)

            with self.assertRaises(VideoProcessingError):
                process_video(
                    input_path,
                    output_path,
                    model,
                    confidence=0.25,
                    max_duration_seconds=30.0,
                )

            model.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()

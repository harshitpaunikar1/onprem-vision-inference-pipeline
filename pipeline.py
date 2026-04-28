"""
On-premises vision inference pipeline with video source handling and result routing.
Supports RTSP, file, and webcam sources with configurable frame processing.
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from inference import InferenceResult, VisionInferenceEngine

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class PipelineConfig:
    source: str = "0"
    frame_skip: int = 3
    max_queue_size: int = 10
    result_callback: Optional[Callable[[InferenceResult], None]] = None
    save_annotated: bool = False
    output_dir: str = "output_frames"
    model_path: str = "model.onnx"
    labels: List[str] = field(default_factory=lambda: ["object"])
    task: str = "detection"
    input_size: tuple = (640, 640)
    conf_threshold: float = 0.45


class VideoSource:
    """Thread-safe video capture with stub frame support."""

    def __init__(self, source: str):
        self.source = source
        self._cap = None
        self._open()

    def _open(self) -> None:
        if not CV2_AVAILABLE:
            return
        try:
            src = int(self.source) if self.source.isdigit() else self.source
            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                logger.warning("Cannot open source: %s", self.source)
                self._cap = None
        except Exception as exc:
            logger.error("VideoCapture error: %s", exc)

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


class ResultAggregator:
    """Accumulates inference results and computes rolling statistics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._results: List[InferenceResult] = []
        self._lock = threading.Lock()

    def add(self, result: InferenceResult) -> None:
        with self._lock:
            self._results.append(result)
            if len(self._results) > self.window_size:
                self._results.pop(0)

    def rolling_stats(self) -> Dict:
        with self._lock:
            if not self._results:
                return {}
            latencies = [r.inference_time_ms for r in self._results]
            detection_counts = [len(r.outputs) for r in self._results]
            return {
                "frames": len(self._results),
                "avg_latency_ms": round(float(np.mean(latencies)), 2),
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
                "avg_fps": round(1000.0 / max(float(np.mean(latencies)), 1), 1),
                "avg_detections": round(float(np.mean(detection_counts)), 2),
            }

    def label_frequency(self) -> Dict[str, int]:
        with self._lock:
            counter: Dict[str, int] = {}
            for r in self._results:
                for det in r.outputs:
                    label = det.get("label", "unknown")
                    counter[label] = counter.get(label, 0) + 1
            return counter


class VisionPipeline:
    """
    Orchestrates video capture, inference, result routing, and statistics collection.
    Supports synchronous and background-threaded operation.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.engine = VisionInferenceEngine(
            model_path=config.model_path,
            labels=config.labels,
            task=config.task,
            input_size=config.input_size,
            conf_threshold=config.conf_threshold,
        )
        self.aggregator = ResultAggregator()
        self._result_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        self._frame_id = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _process_frame(self, frame: np.ndarray) -> InferenceResult:
        return self.engine.predict(frame)

    def _dispatch(self, result: InferenceResult) -> None:
        self.aggregator.add(result)
        try:
            self._result_queue.put_nowait(result)
        except queue.Full:
            self._result_queue.get_nowait()
            self._result_queue.put_nowait(result)
        if self.config.result_callback:
            try:
                self.config.result_callback(result)
            except Exception as exc:
                logger.error("Result callback error: %s", exc)

    def run_sync(self, max_frames: int = 50) -> List[InferenceResult]:
        self.engine.load()
        source = VideoSource(self.config.source)
        results = []
        for _ in range(max_frames * self.config.frame_skip):
            frame = source.read()
            if frame is None:
                break
            self._frame_id += 1
            if self._frame_id % self.config.frame_skip != 0:
                continue
            result = self._process_frame(frame)
            self._dispatch(result)
            results.append(result)
            if len(results) >= max_frames:
                break
        source.release()
        return results

    def _loop(self, max_frames: int) -> None:
        self.engine.load()
        source = VideoSource(self.config.source)
        processed = 0
        while self._running and processed < max_frames:
            frame = source.read()
            if frame is None:
                break
            self._frame_id += 1
            if self._frame_id % self.config.frame_skip != 0:
                continue
            result = self._process_frame(frame)
            self._dispatch(result)
            processed += 1
        source.release()
        self._running = False

    def start(self, max_frames: int = 10000) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(max_frames,), daemon=True)
        self._thread.start()
        logger.info("Vision pipeline started in background.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def get_result(self, timeout: float = 1.0) -> Optional[InferenceResult]:
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stats(self) -> Dict:
        return {
            "frames_seen": self._frame_id,
            "engine": self.engine.stats(),
            "rolling": self.aggregator.rolling_stats(),
            "label_frequency": self.aggregator.label_frequency(),
        }


if __name__ == "__main__":
    def print_result(result: InferenceResult):
        if result.outputs:
            top = result.outputs[0]
            print(f"  Frame: {result.inference_time_ms:.1f}ms | "
                  f"Top: {top.get('label')} ({top.get('confidence', 0):.2f})")

    config = PipelineConfig(
        source="0",
        frame_skip=3,
        model_path="model.onnx",
        labels=["container", "trailer", "truck", "forklift"],
        task="detection",
        conf_threshold=0.45,
        result_callback=print_result,
    )
    pipeline = VisionPipeline(config)
    print("Running 10 frames synchronously...")
    results = pipeline.run_sync(max_frames=10)
    print(f"\nProcessed {len(results)} frames.")
    print("Stats:", pipeline.stats())

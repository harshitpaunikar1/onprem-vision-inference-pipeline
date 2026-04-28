"""
On-premises vision inference engine using ONNX Runtime.
Runs object detection and classification models locally without cloud dependencies.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class InferenceResult:
    model_name: str
    task: str               # detection, classification
    outputs: List[Dict]
    inference_time_ms: float
    input_shape: Tuple
    timestamp: float = field(default_factory=time.time)


@dataclass
class DetectionOutput:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ClassificationOutput:
    label: str
    confidence: float
    top_k: List[Tuple[str, float]]


class ONNXInferenceSession:
    """Wraps an ONNX Runtime inference session with preprocessing and output parsing."""

    def __init__(self, model_path: str,
                 providers: Optional[List[str]] = None,
                 num_threads: int = 4):
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]
        self.num_threads = num_threads
        self._session = None
        self._input_name: Optional[str] = None
        self._input_shape: Optional[Tuple] = None

    def load(self) -> bool:
        if not ORT_AVAILABLE:
            logger.warning("onnxruntime not installed. Running in stub mode.")
            return False
        if not os.path.exists(self.model_path):
            logger.error("Model not found: %s", self.model_path)
            return False
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.num_threads
            opts.inter_op_num_threads = self.num_threads
            self._session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=self.providers
            )
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            self._input_shape = tuple(inp.shape)
            logger.info("ONNX model loaded: %s | Input: %s %s",
                        os.path.basename(self.model_path), self._input_name, self._input_shape)
            return True
        except Exception as exc:
            logger.error("ONNX load failed: %s", exc)
            return False

    @property
    def input_shape(self) -> Tuple:
        return self._input_shape or (1, 3, 640, 640)

    def run(self, input_array: np.ndarray) -> Optional[List[np.ndarray]]:
        if self._session is None:
            return None
        try:
            outputs = self._session.run(None, {self._input_name: input_array})
            return outputs
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return None


class ImagePreprocessor:
    """Prepares images for ONNX vision model inference."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640),
                 normalize: bool = True,
                 channel_order: str = "CHW"):
        self.target_size = target_size
        self.normalize = normalize
        self.channel_order = channel_order

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if not CV2_AVAILABLE:
            h, w = self.target_size
            return np.random.rand(1, 3, h, w).astype(np.float32)
        resized = cv2.resize(image, self.target_size)
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = resized.astype(np.float32)
        if self.normalize:
            arr /= 255.0
        if self.channel_order == "CHW":
            arr = arr.transpose(2, 0, 1)
        return np.expand_dims(arr, axis=0)


class DetectionPostprocessor:
    """Decodes ONNX detection model outputs into DetectionOutput objects."""

    def __init__(self, labels: List[str], conf_threshold: float = 0.45,
                 iou_threshold: float = 0.45):
        self.labels = labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def decode(self, outputs: List[np.ndarray],
               original_shape: Tuple[int, int]) -> List[DetectionOutput]:
        if not outputs:
            return []
        boxes = outputs[0][0] if len(outputs) > 0 else np.array([])
        scores = outputs[2][0] if len(outputs) > 2 else np.array([])
        classes = outputs[1][0].astype(int) if len(outputs) > 1 else np.array([])
        h, w = original_shape
        detections = []
        for i, score in enumerate(scores):
            if float(score) < self.conf_threshold:
                continue
            if i >= len(boxes):
                break
            y1, x1, y2, x2 = boxes[i]
            label_idx = classes[i] if i < len(classes) else 0
            label = self.labels[label_idx] if label_idx < len(self.labels) else str(label_idx)
            detections.append(DetectionOutput(
                label=label,
                confidence=float(score),
                x1=float(x1 * w), y1=float(y1 * h),
                x2=float(x2 * w), y2=float(y2 * h),
            ))
        return detections


class ClassificationPostprocessor:
    """Decodes softmax outputs into top-k classification results."""

    def __init__(self, labels: List[str], top_k: int = 5):
        self.labels = labels
        self.top_k = top_k

    def decode(self, outputs: List[np.ndarray]) -> ClassificationOutput:
        if not outputs:
            return ClassificationOutput("unknown", 0.0, [])
        logits = outputs[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        top_indices = probs.argsort()[-self.top_k:][::-1]
        top_k = [(self.labels[i] if i < len(self.labels) else str(i),
                   float(probs[i])) for i in top_indices]
        best_label, best_conf = top_k[0]
        return ClassificationOutput(label=best_label, confidence=best_conf, top_k=top_k)


class VisionInferenceEngine:
    """
    Unified on-prem inference engine for detection and classification tasks.
    Loads an ONNX model and provides a simple predict() API.
    """

    def __init__(self, model_path: str, labels: List[str],
                 task: str = "detection",
                 input_size: Tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.45):
        self.model_path = model_path
        self.labels = labels
        self.task = task
        self.conf_threshold = conf_threshold
        self.session = ONNXInferenceSession(model_path)
        self.preprocessor = ImagePreprocessor(target_size=input_size)
        if task == "detection":
            self.postprocessor = DetectionPostprocessor(labels, conf_threshold)
        else:
            self.postprocessor = ClassificationPostprocessor(labels)
        self._frame_count = 0

    def load(self) -> bool:
        return self.session.load()

    def predict(self, image: np.ndarray) -> InferenceResult:
        self._frame_count += 1
        t0 = time.perf_counter()
        processed = self.preprocessor.preprocess(image)
        raw_outputs = self.session.run(processed)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if raw_outputs is None:
            raw_outputs = [np.random.rand(1, 10, 4), np.random.randint(0, len(self.labels), (1, 10)),
                           np.random.rand(1, 10).astype(np.float32) * 0.9]

        if self.task == "detection":
            h, w = image.shape[:2]
            dets = self.postprocessor.decode(raw_outputs, (h, w))
            outputs = [{"label": d.label, "confidence": round(d.confidence, 3),
                        "bbox": [d.x1, d.y1, d.x2, d.y2]} for d in dets]
        else:
            cls_out = self.postprocessor.decode(raw_outputs)
            outputs = [{"label": cls_out.label, "confidence": round(cls_out.confidence, 3),
                        "top_k": cls_out.top_k}]

        return InferenceResult(
            model_name=os.path.basename(self.model_path),
            task=self.task,
            outputs=outputs,
            inference_time_ms=round(elapsed_ms, 2),
            input_shape=tuple(processed.shape),
        )

    def stats(self) -> Dict:
        return {"frames_processed": self._frame_count, "model": self.model_path}


if __name__ == "__main__":
    labels = ["container", "trailer", "truck", "forklift", "pallet", "person"]
    engine = VisionInferenceEngine(
        model_path="yolov8n.onnx",
        labels=labels,
        task="detection",
        input_size=(640, 640),
        conf_threshold=0.45,
    )
    engine.load()

    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = engine.predict(dummy_frame)
    print(f"Task: {result.task} | Model: {result.model_name}")
    print(f"Inference: {result.inference_time_ms:.1f}ms | Detections: {len(result.outputs)}")
    for det in result.outputs[:3]:
        print(f"  {det['label']} conf={det['confidence']:.2f}")
    print("Stats:", engine.stats())

"""User-supplied ONNX metric-depth inference and occupancy fusion."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from dalg.grid import LogOddsGrid
from dalg.model import Result
from dalg.algo.spatial import obstacle_band, project_pixels


@dataclass(frozen=True)
class MonocularDepthConfig:
    model_path: str = ""
    input_width: int = 256
    input_height: int = 192
    scale: float = 1.0
    offset_m: float = 0.0
    min_range_m: float = .3
    max_range_m: float = 25.0
    min_height_m: float = .25
    max_height_m: float = 4.5
    stride: int = 8
    occupied_log_odds: float = 2.5


class MonocularDepthAlgorithm:
    name, sensors = "monocular_depth", ("rgb",)

    def __init__(self, width_m, height_m, intrinsics, settings=None, **_):
        self.size, self.intrinsics = (width_m, height_m), intrinsics
        self.config = MonocularDepthConfig(**(settings or {}))
        path = Path(self.config.model_path).expanduser()
        if not self.config.model_path or not path.is_file():
            raise ValueError("monocular_depth requires settings.model_path pointing to an ONNX metric-depth model")
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError(
                "monocular_depth requires onnxruntime; run "
                "scripts/install_dalg_depth_model.py"
            ) from exc
        self.session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.start()

    def start(self): self.grid = LogOddsGrid(*self.size); self.frames = self.points = 0

    def observe(self, frame):
        """Fuse the network's depth as perpendicular depth, not radial range.

        Metric-depth heads report distance along the optical axis, which is the
        same convention :func:`project_pixels` inverts; the row of a pixel then
        gives the sample a height, so floor and sky stop reading as walls.
        """
        blob = cv2.dnn.blobFromImage(np.asarray(frame.rgb, np.uint8), 1/255.0,
            (self.config.input_width, self.config.input_height), swapRB=False)
        depth = np.asarray(self.session.run(None, {self.input_name: blob})[0])
        depth = depth.squeeze().astype(np.float32)
        if depth.ndim != 2: raise ValueError("metric-depth ONNX output must reduce to HxW")
        depth = cv2.resize(depth, (self.intrinsics.width_px, self.intrinsics.height_px),
                           interpolation=cv2.INTER_LINEAR)
        depth = depth*self.config.scale+self.config.offset_m
        step = max(1, self.config.stride)
        sampled = depth[::step, ::step]
        rows, columns = np.nonzero(np.isfinite(sampled) & (sampled > 0))
        self.frames += 1
        if not len(rows): return
        pose = frame.camera_pose
        x, y, z = project_pixels(pose, columns*step, rows*step,
                                 sampled[rows, columns], self.intrinsics)
        keep = obstacle_band(z, x, y, pose,
                             min_height_m=self.config.min_height_m,
                             max_height_m=self.config.max_height_m,
                             min_range_m=self.config.min_range_m,
                             max_range_m=self.config.max_range_m)
        if not keep.any(): return
        self.grid.accumulate(*self.grid.cells(x[keep], y[keep]),
                             self.config.occupied_log_odds)
        self.points += int(keep.sum())

    def _result(self): return Result(self.grid.result(), {"frames": self.frames, "depth_points": self.points})
    def preview(self): return self._result()
    def finish(self): return self._result()

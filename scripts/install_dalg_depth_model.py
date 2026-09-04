#!/usr/bin/env python3
"""Download, export, validate, and install DALG's metric-depth ONNX model."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request


REPOSITORY = "https://github.com/DepthAnything/Depth-Anything-V2.git"
CHECKPOINT_URL = (
    "https://huggingface.co/depth-anything/"
    "Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/"
    "depth_anything_v2_metric_hypersim_vits.pth?download=true"
)
CHECKPOINT_NAME = "depth_anything_v2_metric_hypersim_vits.pth"
WIDTH, HEIGHT = 280, 210
# The upstream requirements also contain matplotlib and open3d for its CLI and
# point-cloud utilities.  Neither is imported by the model exporter, and
# open3d does not publish wheels for every Python version (notably 3.13).
PYTORCH_REQUIREMENTS = ("torch", "torchvision")
EXPORT_REQUIREMENTS = ("opencv-python", "onnx", "onnxruntime")
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading {url}\n       to {destination}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "dvision2-model-installer/1"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("downloaded checkpoint is unexpectedly small")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def export_model(source: Path, checkpoint: Path, output: Path) -> None:
    # Heavy imports happen only in the managed build environment.
    sys.path.insert(0, str(source / "metric_depth"))
    import onnx  # type: ignore[import-not-found]
    import onnxruntime  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from torch import nn
    from depth_anything_v2.dpt import DepthAnythingV2

    class DalgDepthModel(nn.Module):
        """RGB [0, 1] input to metric depth in metres."""

        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            )

        def forward(self, rgb):
            return self.model((rgb - self.mean) / self.std).unsqueeze(1)

    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        max_depth=20.0,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    wrapped = DalgDepthModel(model.eval()).eval()
    sample = torch.zeros(1, 3, HEIGHT, WIDTH, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            sample,
            output,
            input_names=["rgb"],
            output_names=["depth_m"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    import numpy as np
    session = onnxruntime.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    depth = np.asarray(session.run(
        None, {session.get_inputs()[0].name: np.zeros((1, 3, HEIGHT, WIDTH), np.float32)}
    )[0]).squeeze()
    if depth.ndim != 2 or depth.size == 0 or not np.isfinite(depth).all():
        raise RuntimeError(f"unexpected ONNX output shape or values: {depth.shape}")
    print(f"ONNX Runtime validation passed: input 1x3x{HEIGHT}x{WIDTH}, output {depth.shape}")


def normal_mode(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    cache = root / ".cache" / "dalg-depth-model"
    source = cache / "Depth-Anything-V2"
    environment = cache / "venv"
    checkpoint = cache / "checkpoints" / CHECKPOINT_NAME
    destination = Path(args.output).expanduser().resolve() if args.output else (
        root / "assets" / "models" / "depth" / "metric-depth.onnx"
    )

    cache.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        run(["git", "pull", "--ff-only"], cwd=source)
    else:
        run(["git", "clone", "--depth", "1", REPOSITORY, str(source)])

    if not environment.exists():
        run([sys.executable, "-m", "venv", str(environment)])
    build_python = environment / "bin" / "python"
    # Export is CPU-only.  PyPI's Linux torch package can pull several GB of
    # CUDA libraries, so deliberately use PyTorch's official CPU wheel index.
    run([str(build_python), "-m", "pip", "install", *PYTORCH_REQUIREMENTS,
         "--index-url", PYTORCH_CPU_INDEX])
    run([str(build_python), "-m", "pip", "install", *EXPORT_REQUIREMENTS])
    if importlib.util.find_spec("onnxruntime") is None:
        print("Installing DALG's ONNX Runtime dependency", flush=True)
        run([sys.executable, "-m", "pip", "install", "onnxruntime"])

    if args.force_download or not checkpoint.is_file():
        download(CHECKPOINT_URL, checkpoint)
    else:
        print(f"Reusing checkpoint {checkpoint}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, candidate_name = tempfile.mkstemp(
        prefix=".metric-depth-", suffix=".onnx", dir=destination.parent
    )
    os.close(fd)
    candidate = Path(candidate_name)
    candidate.unlink()
    try:
        run([
            str(build_python), str(Path(__file__).resolve()), "--export-internal",
            "--source", str(source), "--checkpoint", str(checkpoint),
            "--output", str(candidate),
        ])
        os.replace(candidate, destination)
    finally:
        candidate.unlink(missing_ok=True)
    print(f"Installed DALG metric-depth model at {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="model destination (defaults to assets/models/depth)")
    parser.add_argument(
        "--force-download", action="store_true",
        help="download the checkpoint again instead of using the cached copy",
    )
    parser.add_argument("--export-internal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.export_internal:
        if not args.source or not args.checkpoint or not args.output:
            raise SystemExit("internal export requires source, checkpoint, and output")
        export_model(args.source, args.checkpoint, Path(args.output))
    else:
        normal_mode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

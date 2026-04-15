#!/usr/bin/env python3
"""
Export PyLinkx game models from .zip (PyTorch/sb3) to .onnx (onnxruntime).

Reads models/easy_model.zip, models/medium_model.zip, models/hard_model.zip
and writes the ONNX equivalents to src/models/.

Requires dev deps (torch, sb3-contrib). Run once after updating game models:
    uv run python scripts/export_onnx.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO

GAME_MODELS = ["easy_model", "medium_model", "hard_model"]
SRC_DIR = Path("models")
DST_DIR = Path("src/models")


class PolicyExporter(nn.Module):
    """Thin wrapper that traces the inference path: (grid, scalars) -> logits."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, grid, scalars):
        obs = {"grid": grid, "scalars": scalars}
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        return self.policy.action_net(latent_pi)  # (1, 6) raw logits


def export(name: str) -> None:
    src = SRC_DIR / f"{name}.zip"
    dst = DST_DIR / f"{name}.onnx"

    if not src.exists():
        print(f"  SKIP {name}: {src} not found")
        return

    model = MaskablePPO.load(src, device="cpu")
    model.policy.set_training_mode(False)

    exporter = PolicyExporter(model.policy)
    exporter.eval()

    dummy_grid = torch.zeros(1, 9, 9, 1)
    dummy_scalars = torch.zeros(1, 258)

    torch.onnx.export(
        exporter,
        (dummy_grid, dummy_scalars),
        str(dst),
        input_names=["grid", "scalars"],
        output_names=["logits"],
        dynamic_axes={
            "grid": {0: "batch"},
            "scalars": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,  # TorchScript exporter — produces a single .onnx file (no .data sidecar)
    )

    # Validate: ONNX output must match PyTorch output within tolerance
    import onnxruntime as ort

    rng = np.random.default_rng(42)
    val_grid = rng.random((1, 9, 9, 1), dtype=np.float32)
    val_scalars = rng.random((1, 258), dtype=np.float32)

    with torch.no_grad():
        pt_logits = exporter(
            torch.from_numpy(val_grid), torch.from_numpy(val_scalars)
        ).numpy()

    session = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    ort_logits = session.run(["logits"], {"grid": val_grid, "scalars": val_scalars})[0]

    if not np.allclose(pt_logits, ort_logits, atol=1e-5):
        raise RuntimeError(
            f"{name}: ONNX output does not match PyTorch output\n"
            f"  PyTorch: {pt_logits}\n"
            f"  ONNX:    {ort_logits}"
        )

    print(f"  \u2713 {name:15s} -> {dst}")


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(GAME_MODELS)} models from {SRC_DIR}/ to {DST_DIR}/")
    for name in GAME_MODELS:
        export(name)
    print("Done.")


if __name__ == "__main__":
    main()

"""Export trained model to ONNX for web demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model import load_model_from_checkpoint


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 18,
) -> Path:
    model = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))

    dummy_input = torch.randn(1, 1, 48, 48)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset_version,
    )

    print(f"ONNX model saved to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_cnn.pt"))
    parser.add_argument("--output", type=Path, default=Path("demo/model.onnx"))
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output)


if __name__ == "__main__":
    main()

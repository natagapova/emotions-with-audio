"""Post-training quantization and Core ML export."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from src.dataset import create_dataloaders
from src.evaluate import collect_predictions, load_model
from src.model import create_model, load_model_from_checkpoint
from src.train import get_device
from sklearn.metrics import accuracy_score


def model_size_mb(path: Path) -> float:
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total = path.stat().st_size
    return total / (1024 * 1024)


def measure_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple[int, ...] = (1, 1, 48, 48),
    runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float]:
    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def export_coreml(
    model: nn.Module,
    output_path: Path,
    quantize: bool = False,
) -> Path:
    model_cpu = model.cpu().eval()
    example_input = torch.randn(1, 1, 48, 48)

    traced = torch.jit.trace(model_cpu, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=example_input.shape)],
        convert_to="mlprogram",
    )

    if quantize:
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )

        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512),
        )
        mlmodel = linear_quantize_weights(mlmodel, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return output_path


def quantize_and_compare(
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
) -> dict:
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = checkpoint.get("arch", "cnn")

    fp32_model = load_model(checkpoint_path, device)
    loaders = create_dataloaders(
        data_dir, batch_size=batch_size, use_weighted_sampler=False, num_workers=0
    )

    y_true, y_pred_fp32 = collect_predictions(fp32_model, loaders["test"], device)
    fp32_accuracy = accuracy_score(y_true, y_pred_fp32)

    fp32_size = model_size_mb(checkpoint_path)
    fp32_latency_mean, fp32_latency_std = measure_latency(fp32_model, device)

    # INT8 dynamic quantization (Linear layers only)
    fp32_cpu = create_model(arch).cpu()
    fp32_cpu.load_state_dict(fp32_model.cpu().state_dict())
    fp32_cpu.eval()

    supported_engines = torch.backends.quantized.supported_engines
    if "qnnpack" in supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif supported_engines:
        torch.backends.quantized.engine = supported_engines[0]

    int8_model = torch.ao.quantization.quantize_dynamic(
        fp32_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    int8_model.eval()

    _, y_pred_int8 = collect_predictions(int8_model, loaders["test"], torch.device("cpu"))
    int8_accuracy = accuracy_score(y_true, y_pred_int8)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save(int8_model.state_dict(), tmp.name)
        int8_size = model_size_mb(Path(tmp.name))

    int8_latency_mean, int8_latency_std = measure_latency(int8_model, torch.device("cpu"))

    fp32_mlpackage = output_dir / f"emotion_{arch}_fp32.mlpackage"
    int8_mlpackage = output_dir / f"emotion_{arch}_int8.mlpackage"
    export_coreml(fp32_cpu, fp32_mlpackage, quantize=False)
    export_coreml(fp32_cpu, int8_mlpackage, quantize=True)

    accuracy_drop = (fp32_accuracy - int8_accuracy) * 100

    report = {
        "arch": arch,
        "fp32_accuracy": fp32_accuracy,
        "int8_accuracy": int8_accuracy,
        "accuracy_drop_pp": accuracy_drop,
        "fp32_size_mb": fp32_size,
        "int8_size_mb": int8_size,
        "fp32_latency_ms_mean": fp32_latency_mean,
        "fp32_latency_ms_std": fp32_latency_std,
        "int8_latency_ms_mean": int8_latency_mean,
        "int8_latency_ms_std": int8_latency_std,
        "coreml_fp32_path": str(fp32_mlpackage),
        "coreml_int8_path": str(int8_mlpackage),
    }

    report_path = output_dir / f"quantization_report_{arch}.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"FP32 accuracy: {fp32_accuracy:.4f} | INT8 accuracy: {int8_accuracy:.4f}")
    print(f"Accuracy drop: {accuracy_drop:.2f} pp")
    print(f"FP32 size: {fp32_size:.2f} MB | INT8 size: {int8_size:.2f} MB")
    print(
        f"FP32 latency: {fp32_latency_mean:.2f} ± {fp32_latency_std:.2f} ms | "
        f"INT8 latency: {int8_latency_mean:.2f} ± {int8_latency_std:.2f} ms"
    )
    print(f"Report saved to {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize and export to Core ML")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_cnn.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    quantize_and_compare(args.checkpoint, args.data_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()

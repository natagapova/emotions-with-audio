"""Tests for quantization accuracy drop."""

import json
from pathlib import Path

import pytest


REPORT_PATH = Path("models/quantization_report.json")
MAX_ACCURACY_DROP_PP = 3.0


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT_PATH.exists():
        pytest.skip(
            "Quantization report not found. Run: python src/quantize.py"
        )
    with REPORT_PATH.open() as f:
        return json.load(f)


def test_accuracy_drop_within_threshold(report: dict) -> None:
    drop = report["accuracy_drop_pp"]
    assert drop < MAX_ACCURACY_DROP_PP, (
        f"INT8 accuracy drop {drop:.2f} pp exceeds {MAX_ACCURACY_DROP_PP} pp threshold"
    )


def test_int8_not_worse_than_fp32(report: dict) -> None:
    assert report["int8_accuracy"] <= report["fp32_accuracy"] + 0.01


def test_size_reduction(report: dict) -> None:
    assert report["int8_size_mb"] <= report["fp32_size_mb"]

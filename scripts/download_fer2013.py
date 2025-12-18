"""Download FER2013 from HuggingFace and save as CSV splits in data/."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

SPLIT_MAP = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}


def main(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading icml_face_data.csv (FER2013) from HuggingFace...")
    csv_path = hf_hub_download(
        "DerrickUnleashed/FER-2013",
        "icml_face_data.csv",
        repo_type="dataset",
    )

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    if len(df) != 35887:
        raise ValueError(
            f"Expected 35887 samples, got {len(df)}. "
            "Dataset may have changed — stop and verify splits."
        )

    if "Usage" not in df.columns:
        raise ValueError(f"Missing Usage column. Columns: {df.columns.tolist()}")

    for usage, split in SPLIT_MAP.items():
        split_df = df[df["Usage"] == usage][["emotion", "pixels"]].reset_index(drop=True)
        out_path = data_dir / f"fer2013_{split}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"  {split}: {len(split_df)} samples -> {out_path}")

    print("\nClass distribution (train):")
    train_df = df[df["Usage"] == "Training"]
    counts = train_df["emotion"].value_counts().sort_index()
    for idx, count in counts.items():
        label = EMOTION_LABELS[idx]
        print(f"  {label:10s} ({idx}): {count:5d} ({100 * count / len(train_df):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FER2013 dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save CSV splits",
    )
    args = parser.parse_args()
    main(args.data_dir)

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tabular_csv(tmp_path: Path):
    def create_csv(name: str = "dataset.csv", task: str = "regression") -> Path:
        path = tmp_path / name
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["feature_a", "feature_b", "target"])
            for index in range(30):
                feature_a = index / 5
                feature_b = (index * 3) % 11
                if task == "classification":
                    target = "odd" if index % 2 else "even"
                else:
                    target = 0.5 * feature_a - 0.25 * feature_b + 2
                writer.writerow([feature_a, feature_b, target])
        return path

    return create_csv


@pytest.fixture
def image_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "images"
    for class_index, class_name in enumerate(("blue", "orange")):
        class_dir = dataset / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(6):
            image = Image.new(
                "RGB",
                (16, 16),
                (20, 40 + class_index * 180, 90 + image_index),
            )
            image.save(class_dir / f"{image_index}.png")
    return dataset

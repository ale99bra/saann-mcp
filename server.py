"""MCP tools for training and evaluating SaANN models on local datasets."""

from __future__ import annotations

import json
import math
import sys
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

with redirect_stdout(sys.stderr):
    from saann import losses as saann_losses
    from saann.metrics import Metrics
    from saann.models import CNN, SequentialModel, load_model
    from saann.processing import ImageProcessing


mcp = MCPServer("SaANN Local ML Engine")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CHECKPOINT_DIRECTORY = "saann_models"


def _redirect_saann_stdout(function: Any) -> Any:
    """Keep library print output off the MCP stdio protocol stream."""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with redirect_stdout(sys.stderr):
            return function(*args, **kwargs)

    return wrapped


def _expose_expected_tool_errors(function: Any) -> Any:
    """Return expected input and file errors to MCP clients without hiding details."""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except (FileNotFoundError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    return wrapped


def _read_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV dataset and provide actionable errors for common input issues."""
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV dataset does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV dataset is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV dataset is malformed: {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV dataset is not valid UTF-8 text: {path}") from exc
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("CSV dataset must contain at least one row and a target column after its features.")
    if frame.columns.has_duplicates:
        raise ValueError("CSV dataset must have unique column names.")
    return frame


def _read_tabular_data(file_path: str) -> tuple[np.ndarray, np.ndarray, bool, list[str], list[Any]]:
    """Load numeric features and prepare a regression or classification target."""
    frame = _read_csv(file_path)
    feature_columns = [str(column) for column in frame.columns[:-1]]
    target_column = frame.columns[-1]

    try:
        features = frame.iloc[:, :-1].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Every feature column in the CSV must contain numeric values.") from exc
    if not np.isfinite(features).all():
        raise ValueError("Feature columns contain missing or non-finite values.")

    target = frame.iloc[:, -1]
    if target.isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")
    numeric_target = pd.to_numeric(target, errors="coerce")
    is_numeric = numeric_target.notna().all()
    is_integer_target = is_numeric and np.equal(
        numeric_target.to_numpy(dtype=np.float64),
        np.floor(numeric_target.to_numpy(dtype=np.float64)),
    ).all()
    classification = not is_numeric or (
        is_integer_target and 1 < target.nunique() <= 20
    )

    if classification:
        encoded, labels = pd.factorize(target.astype(str), sort=True)
        if len(labels) < 2:
            raise ValueError("Classification requires at least two distinct target labels.")
        targets = np.eye(len(labels), dtype=np.float64)[encoded]
        class_labels = [str(value) for value in labels]
    else:
        target_values = numeric_target.to_numpy(dtype=np.float64)
        if not np.isfinite(target_values).all():
            raise ValueError(f"Target column '{target_column}' contains non-finite values.")
        targets = target_values.reshape(-1, 1)
        class_labels = []

    return features, targets, classification, feature_columns, class_labels


def _split_data(
    features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make a deterministic 80/20 split while retaining at least one training row."""
    if len(features) < 2:
        raise ValueError("Training requires at least two dataset rows.")
    indices = np.random.default_rng(42).permutation(len(features))
    test_size = max(1, math.ceil(len(features) * 0.2))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return (
        features[train_indices],
        features[test_indices],
        targets[train_indices],
        targets[test_indices],
    )


def _validate_training_parameters(epochs: int, lr: float, batch_size: int) -> None:
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a finite number greater than 0.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")


def _validate_image_directory(directory_path: str) -> tuple[Path, list[Path]]:
    """Check that an image dataset contains class subdirectories with images."""
    path = Path(directory_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Image dataset directory does not exist: {path}")
    class_directories = sorted(
        child for child in path.iterdir() if child.is_dir() and child.name != "resized"
    )
    if len(class_directories) < 2:
        raise ValueError("CNN image datasets must contain at least two class subdirectories.")
    populated_classes = [
        child
        for child in class_directories
        if any(image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS for image in child.iterdir())
    ]
    if len(populated_classes) != len(class_directories):
        raise ValueError("Every image class subdirectory must contain at least one supported image.")
    return path, class_directories


def _saann_image_class_names(class_directories: list[Path]) -> list[str]:
    """Apply SaANN's underscore-prefix convention to image directory names."""
    class_names = [directory.name.split("_", maxsplit=1)[0] for directory in class_directories]
    if any(not name for name in class_names):
        raise ValueError("SaANN image class names must have a non-empty prefix before '_'.")
    if len(set(class_names)) != len(class_names):
        raise ValueError(
            "SaANN treats image class directories with the same prefix before '_' as one class."
        )
    return class_names


def _checkpoint_path() -> Path:
    """Create a unique checkpoint path under the current working directory."""
    directory = Path.cwd() / CHECKPOINT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"saann-{timestamp}-{uuid.uuid4().hex[:8]}.saann"


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".json")


def _write_metadata(model_path: Path, metadata: dict[str, Any]) -> None:
    with _metadata_path(model_path).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)


def _read_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = _metadata_path(model_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}. "
            "Evaluation requires a checkpoint created by this MCP server."
        )
    try:
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Model metadata is malformed: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Model metadata must be a JSON object: {metadata_path}")
    return metadata


def _evaluation_metrics(
    targets: np.ndarray, predictions: np.ndarray, classification: bool
) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Model output shape {predictions.shape} does not match target shape {targets.shape}."
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")
    loss_name = "cross-entropy" if classification else "mse"
    loss_function, _, _ = saann_losses.get_loss_functions(loss_name)
    metrics = {"loss": float(loss_function(targets, predictions))}
    if classification:
        metrics["accuracy"] = float(Metrics(targets, predictions).micro_f1())
    return metrics


@mcp.tool()
@_expose_expected_tool_errors
def inspect_dataset(file_path: str) -> dict[str, Any]:
    """Inspect a local CSV or class-folder image dataset before model training.

    CSV files are summarized by shape, columns, missing-value counts, and
    descriptive statistics. For CNN image directories, each immediate
    subdirectory is treated as a class and the image count is reported.
    """
    path = Path(file_path).expanduser()
    if path.is_dir():
        path, class_directories = _validate_image_directory(str(path))
        counts = {
            directory.name: sum(
                image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
                for image in directory.iterdir()
            )
            for directory in class_directories
        }
        return {
            "path": str(path.resolve()),
            "format": "image_directory",
            "shape": [sum(counts.values()), len(counts)],
            "columns": [directory.name for directory in class_directories],
            "missing_values": {name: 0 for name in counts},
            "summary_statistics": {"images_per_class": counts},
        }

    frame = _read_csv(str(path))
    return {
        "path": str(path.resolve()),
        "format": "csv",
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [str(column) for column in frame.columns],
        "missing_values": {str(column): int(count) for column, count in frame.isna().sum().items()},
        "summary_statistics": json.loads(frame.describe(include="all").to_json()),
    }


@mcp.tool()
@_redirect_saann_stdout
@_expose_expected_tool_errors
def train_model(
    dataset_path: str,
    model_type: str = "mlp",
    epochs: int = 20,
    lr: float = 0.01,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Train a SaANN MLP on a CSV or CNN on a class-folder image directory.

    For MLP training, the CSV's last column is the target and all preceding
    columns must be numeric features. Integer targets with up to 20 distinct
    values and non-numeric targets are treated as classification labels; other
    numeric targets are treated as regression. CNN datasets must contain one
    subdirectory per class with supported image files. The SaANN model and its
    evaluation metadata are saved locally under ./saann_models/.
    """
    _validate_training_parameters(epochs, lr, batch_size)
    if model_type not in {"mlp", "cnn"}:
        raise ValueError("model_type must be either 'mlp' or 'cnn'.")

    if model_type == "mlp":
        features, targets, classification, feature_columns, class_labels = _read_tabular_data(
            dataset_path
        )
        X_train, X_test, y_train, y_test = _split_data(features, targets)
        output_dim = y_train.shape[1]
        hidden_dim = min(128, max(16, features.shape[1] * 2))
        model = SequentialModel(gpu=False)
        model.construct(
            layers_info=[
                (features.shape[1], hidden_dim, "relu", "he"),
                (
                    hidden_dim,
                    output_dim,
                    "softmax" if classification else "linear",
                    "he",
                ),
            ],
            learning_rate=lr,
        )
        loss_function = "cross-entropy" if classification else "mse"
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            loss_function=loss_function,
            graphical=False,
        )
        predictions = model.predict(X_test)
        metrics = _evaluation_metrics(y_test, predictions, classification)
        metadata = {
            "model_type": model_type,
            "task": "classification" if classification else "regression",
            "feature_columns": feature_columns,
            "class_labels": class_labels,
        }
    else:
        image_path, class_directories = _validate_image_directory(dataset_path)
        expected_classes = _saann_image_class_names(class_directories)
        image_processor = ImageProcessing(images_path=str(image_path))
        X_train, X_test, y_train, y_test, image_classes = image_processor.ready_dataset(
            size=48,
            shuffle=True,
            split_test_percentage=0.2,
        )
        if set(image_classes) != set(expected_classes):
            raise ValueError(
                f"SaANN image classes {image_classes} did not match the normalized "
                f"dataset classes {expected_classes}."
            )
        model = CNN(filter_size=3, num_filters=8, padding=1, stride=1, gpu=False)
        input_size = model.get_input_size(X_train=X_train)
        output_dim = y_train.shape[1]
        hidden_dim = min(128, max(16, input_size * 2))
        model.construct(
            layers_info=[
                (input_size, hidden_dim, "relu", "he"),
                (hidden_dim, output_dim, "softmax", "he"),
            ],
            learning_rate=lr,
        )
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            graphical=False,
        )
        predictions = model.predict(X_test)
        metrics = _evaluation_metrics(y_test, predictions, classification=True)
        metadata = {
            "model_type": model_type,
            "task": "classification",
            "class_labels": list(image_classes),
        }

    train_loss = float(model.final_loss)
    if not math.isfinite(train_loss):
        raise ValueError("Training produced a non-finite loss.")
    metrics["train_loss"] = train_loss

    checkpoint = _checkpoint_path()
    model.save_model(str(checkpoint))
    _write_metadata(checkpoint, metadata)
    return {
        "status": "completed",
        "model_type": model_type,
        "checkpoint_path": str(checkpoint.resolve()),
        "metrics": metrics,
    }


@mcp.tool()
@_redirect_saann_stdout
@_expose_expected_tool_errors
def evaluate_model(model_path: str, test_data_path: str) -> dict[str, Any]:
    """Evaluate an MCP-created SaANN checkpoint against a local test dataset.

    The test CSV must have the same feature columns and target format as the
    training CSV. CNN test data must use the same class-folder names as the
    training image directory. Returns SaANN's native MSE or cross-entropy loss,
    plus its micro-F1 score (single-label accuracy) for classification.
    """
    checkpoint = Path(model_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SaANN checkpoint does not exist: {checkpoint}")
    metadata = _read_metadata(checkpoint)
    model = load_model(str(checkpoint))

    if metadata.get("model_type") == "mlp":
        features, targets, classification, feature_columns, class_labels = _read_tabular_data(
            test_data_path
        )
        if feature_columns != metadata.get("feature_columns"):
            raise ValueError("Test CSV feature columns do not match those used to train the model.")
        if (metadata.get("task") == "classification") != classification:
            raise ValueError("Test CSV target type does not match the model's training task.")
        if classification:
            if class_labels != metadata.get("class_labels"):
                raise ValueError("Test CSV target labels do not match the model's training labels.")
    elif metadata.get("model_type") == "cnn":
        _, class_directories = _validate_image_directory(test_data_path)
        expected_classes = metadata.get("class_labels")
        if not isinstance(expected_classes, list) or not all(
            isinstance(name, str) for name in expected_classes
        ):
            raise ValueError("CNN checkpoint metadata has invalid class labels.")
        observed_classes = _saann_image_class_names(class_directories)
        if set(observed_classes) != set(expected_classes):
            raise ValueError("Test image classes do not match the model's training classes.")
        image_processor = ImageProcessing(images_path=str(Path(test_data_path).expanduser()))
        features, targets, class_names = image_processor.ready_dataset(
            size=48,
            shuffle=False,
        )
        if set(class_names) != set(expected_classes):
            raise ValueError("SaANN's test image classes do not match the training classes.")
        if list(class_names) != expected_classes:
            targets = targets[:, [class_names.index(name) for name in expected_classes]]
        classification = True
    else:
        raise ValueError("Checkpoint metadata has an unsupported SaANN model_type.")

    predictions = model.predict(features)
    return {
        "status": "completed",
        "model_path": str(checkpoint.resolve()),
        "metrics": _evaluation_metrics(targets, predictions, classification),
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from server import _evaluation_metrics, _read_tabular_data, mcp


def _result_json(result) -> dict:
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        content, structured = result
        if isinstance(structured, dict):
            return structured
        if content and hasattr(content[0], "text"):
            return json.loads(content[0].text)
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    content = getattr(result, "content", None)
    if content and hasattr(content[0], "text"):
        return json.loads(content[0].text)
    raise AssertionError(f"Unexpected FastMCP tool result: {result!r}")


async def _assert_tool_error(name: str, arguments: dict, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        await mcp.call_tool(name, arguments)


def _input_schema(tool) -> dict:
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema")
    return schema


@pytest.mark.asyncio
async def test_tool_discovery_exposes_documented_schemas():
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    assert set(tools) == {"inspect_dataset", "train_model", "evaluate_model"}
    assert all(tool.description for tool in tools.values())

    inspect_schema = _input_schema(tools["inspect_dataset"])
    assert set(inspect_schema["properties"]) == {"file_path"}
    assert inspect_schema["properties"]["file_path"]["type"] == "string"
    assert inspect_schema["required"] == ["file_path"]

    training_schema = _input_schema(tools["train_model"])
    assert set(training_schema["properties"]) == {
        "dataset_path",
        "model_type",
        "epochs",
        "lr",
        "batch_size",
    }
    assert training_schema["properties"]["model_type"]["type"] == "string"
    assert training_schema["properties"]["epochs"]["type"] == "integer"
    assert training_schema["properties"]["lr"]["type"] == "number"
    assert training_schema["properties"]["batch_size"]["type"] == "integer"
    assert training_schema["required"] == ["dataset_path"]

    evaluation_schema = _input_schema(tools["evaluate_model"])
    assert set(evaluation_schema["properties"]) == {"model_path", "test_data_path"}
    assert evaluation_schema["required"] == ["model_path", "test_data_path"]


@pytest.mark.asyncio
async def test_inspect_train_and_evaluate_csv_workflow(
    tabular_csv, tmp_path, monkeypatch
):
    dataset = tabular_csv()
    monkeypatch.chdir(tmp_path)

    inspection_result = await mcp.call_tool(
        "inspect_dataset", {"file_path": str(dataset)}
    )
    training_result = await mcp.call_tool(
        "train_model",
        {
            "dataset_path": str(dataset),
            "model_type": "mlp",
            "epochs": 1,
            "lr": 0.01,
            "batch_size": 8,
        },
    )
    training = _result_json(training_result)
    mismatched_data = tmp_path / "mismatched.csv"
    mismatched_data.write_text(
        dataset.read_text(encoding="utf-8").replace("feature_a", "other_feature"),
        encoding="utf-8",
    )
    await _assert_tool_error(
        "evaluate_model",
        {
            "model_path": training["checkpoint_path"],
            "test_data_path": str(mismatched_data),
        },
        "feature columns do not match",
    )
    evaluation_result = await mcp.call_tool(
        "evaluate_model",
        {
            "model_path": training["checkpoint_path"],
            "test_data_path": str(dataset),
        },
    )

    inspection = _result_json(inspection_result)
    evaluation = _result_json(evaluation_result)
    assert inspection["format"] == "csv"
    assert inspection["shape"] == [30, 3]
    assert training["status"] == "completed"
    assert Path(training["checkpoint_path"]).is_file()
    assert Path(training["checkpoint_path"] + ".json").is_file()
    assert math.isfinite(training["metrics"]["train_loss"])
    assert evaluation["status"] == "completed"
    assert math.isfinite(evaluation["metrics"]["loss"])


@pytest.mark.asyncio
async def test_mcp_trains_and_evaluates_classification_csv(
    tabular_csv, tmp_path, monkeypatch
):
    dataset = tabular_csv(task="classification")
    monkeypatch.chdir(tmp_path)

    training_result = await mcp.call_tool(
        "train_model",
        {
            "dataset_path": str(dataset),
            "model_type": "mlp",
            "epochs": 1,
            "lr": 0.01,
            "batch_size": 8,
        },
    )
    training = _result_json(training_result)
    evaluation_result = await mcp.call_tool(
        "evaluate_model",
        {
            "model_path": training["checkpoint_path"],
            "test_data_path": str(dataset),
        },
    )

    evaluation = _result_json(evaluation_result)
    assert training["status"] == "completed"
    assert "accuracy" in training["metrics"]
    assert "accuracy" in evaluation["metrics"]
    assert 0.0 <= evaluation["metrics"]["accuracy"] <= 1.0


@pytest.mark.asyncio
async def test_mcp_trains_cnn_from_temporary_class_directories(
    image_dataset, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    inspection_result = await mcp.call_tool(
        "inspect_dataset", {"file_path": str(image_dataset)}
    )
    training_result = await mcp.call_tool(
        "train_model",
        {
            "dataset_path": str(image_dataset),
            "model_type": "cnn",
            "epochs": 1,
            "batch_size": 2,
        },
    )
    training = _result_json(training_result)
    evaluation_result = await mcp.call_tool(
        "evaluate_model",
        {
            "model_path": training["checkpoint_path"],
            "test_data_path": str(image_dataset),
        },
    )

    inspection = _result_json(inspection_result)
    evaluation = _result_json(evaluation_result)
    assert inspection["format"] == "image_directory"
    assert inspection["shape"] == [12, 2]
    assert training["status"] == "completed"
    assert math.isfinite(training["metrics"]["train_loss"])
    assert 0.0 <= training["metrics"]["accuracy"] <= 1.0
    assert evaluation["status"] == "completed"
    assert 0.0 <= evaluation["metrics"]["accuracy"] <= 1.0


@pytest.mark.asyncio
async def test_mcp_training_rejects_missing_files_and_invalid_parameters(tmp_path):
    await _assert_tool_error(
        "train_model",
        {"dataset_path": str(tmp_path / "not-found.csv"), "epochs": 1},
        "does not exist",
    )
    await _assert_tool_error(
        "train_model",
        {
            "dataset_path": str(tmp_path / "not-found.csv"),
            "model_type": "rnn",
            "epochs": 1,
        },
        "model_type must be either",
    )
    await _assert_tool_error(
        "evaluate_model",
        {
            "model_path": str(tmp_path / "missing.saann"),
            "test_data_path": str(tmp_path / "test.csv"),
        },
        "checkpoint does not exist",
    )


def test_tabular_validation_rejects_non_numeric_features_and_malformed_csv(tmp_path):
    non_numeric = tmp_path / "nonnumeric.csv"
    non_numeric.write_text("feature,target\nnot-a-number,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="numeric values"):
        _read_tabular_data(str(non_numeric))

    malformed = tmp_path / "malformed.csv"
    malformed.write_text('a,b\n1,"2\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        _read_tabular_data(str(malformed))


def test_evaluation_metrics_validate_prediction_dimensions():
    targets = np.ones((2, 1))
    with pytest.raises(ValueError, match="does not match target shape"):
        _evaluation_metrics(targets, np.ones((2, 2)), classification=False)

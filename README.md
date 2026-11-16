# SaANN MCP

[![Tests](https://github.com/ale99bra/SaANN/workflows/Run%20Tests/badge.svg)](https://github.com/ale99bra/SaANN/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A standalone, local [Model Context Protocol](https://modelcontextprotocol.io/)
server that exposes the [SaANN](https://github.com/ale99bra/SaANN/) machine-learning library over MCP stdio. It
trains and evaluates models on the machine running the client; it requires no
Docker, FastAPI service, or external backend.

## Requirements and installation

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

From the repository root, synchronize the project environment:

```bash
uv sync
```

This installs the MCP server, SaANN, and the development/test dependencies.
SaANN is installed from its upstream Git repository as declared in
`pyproject.toml`.

## Run and test

Start the MCP server over stdio:

```bash
uv run saann-mcp
```

Run the test suite:

```bash
uv run pytest -v
```

The tests use temporary CSVs and image folders. Training checkpoints are
written under `saann_models/` in the server's current working directory.

## MCP tools

- `inspect_dataset(file_path)`: Inspect a CSV or an image directory organized
  into one subdirectory per class.
- `train_model(dataset_path, model_type="mlp", epochs=20, lr=0.01,
  batch_size=32)`: Train an MLP from a CSV or a CNN from a class-folder image
  directory.
- `evaluate_model(model_path, test_data_path)`: Evaluate a checkpoint created
  by this server. The corresponding `.saann.json` metadata file must remain
  beside the checkpoint. Evaluation reports SaANN's native loss and, for
  classification, its micro-F1 score (equivalent to single-label accuracy).

For MLP datasets, use numeric feature columns and a target in the final column.
Integer targets with up to 20 distinct values and non-numeric targets are
treated as classification; other numeric targets are treated as regression.
For CNN datasets, include at least two class subdirectories containing BMP,
JPEG, PNG, TIFF, or WebP images. Evaluation data must use the same features or
classes used during training. SaANN uses the part of each class-directory name
before the first underscore as its class label (for example, `green_area` is
treated as `green`), so these normalized class names must be unique.

## Configure an MCP client

Replace the example path with the absolute path to this repository.

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "saann": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\path\\to\\saann-mcp",
        "saann-mcp"
      ]
    }
  }
}
```

### Cursor (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "saann": {
      "command": "uvx",
      "args": [
        "--from",
        "C:\\path\\to\\saann-mcp",
        "saann-mcp"
      ]
    }
  }
}
```

Both configurations run the local project. `uv run` synchronizes the
repository's project environment; `uvx --from` runs the project command from
the specified local source.

## License

This project is licensed under the MIT License - see the [LICENSE](/LICENCE) file for details.

## Author

**Alessio Branda** - [GitHub Profile](https://github.com/ale99bra)

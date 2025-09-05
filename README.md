# MA-POCA Implementation for RLlib

This project provides an implementation of the **MA-POCA (Multi-Agent Posthumous Credit Assignment)** algorithm, designed to work within the Ray RLlib framework. The implementation is based on the original paper and is demonstrated on a multi-agent environment from the PettingZoo library.

## Core Technologies

-   [Ray (RLlib)](https://www.ray.io/): For distributed reinforcement learning.
-   [PyTorch](https://pytorch.org/): As the deep learning backend.
-   [MLFlow](https://mlflow.org/): For experiment tracking.
-   [PettingZoo](https://pettingzoo.farama.org/): For multi-agent environments.

## Project Structure

-   `.github/`: Contains GitHub Actions workflows and templates.
-   `docs/`: Contains project documentation.
-   `src/ma_poca/`: Contains the source code for the MA-POCA algorithm, including the custom policy and model with a self-attention critic.
-   `examples/`: Contains example scripts to train the MA-POCA agent on PettingZoo environments.
-   `tests/`: Contains tests for the project.

## Development

### Setup

To set up the development environment, first activate the conda environment and then install the project in editable mode. This will also install all the required dependencies from `pyproject.toml`.

```bash
conda activate JulesRayMARL
pip install -e .
```

To install development dependencies (like `pytest`), run:
```bash
pip install -e .[dev]
```

### Code Quality

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and code formatting. Ruff is a fast Python linter and formatter written in Rust.

To check for linting errors:
```bash
ruff check .
```

To automatically fix linting errors:
```bash
ruff check . --fix
```

To format code:
```bash
ruff format .
```

### Usage

To run the example training script for MA-POCA on the `simple_spread_v3` environment:

```bash
python -m examples.train_mpe
```

To run the hyperparameter tuning example:

```bash
python -m examples.hpo_mpe
```

These will start the training process. Metrics and results will be logged to the console and also saved to an `mlruns` directory, which can be viewed with the MLflow UI:

```bash
mlflow ui
```

### Design Documentation

#### Episode Reward Mean Retrieval

In the MA-POCA implementation, the episode reward mean is obtained from the training results using the following key path:
- `result["env_runners"]["episode_reward_mean"]`

This is important to note as the key structure has changed from previous versions of RLlib where `result["episode_reward_mean"]` was used. The updated key path ensures correct retrieval of the episode reward mean from the training results.

## Troubleshooting

If you encounter any issues, please check the following:

1. Ensure all dependencies are installed correctly.
2. Check that the PettingZoo environment is compatible with the version specified in `pyproject.toml`.
3. Verify that the Ray cluster is properly configured if running in distributed mode.
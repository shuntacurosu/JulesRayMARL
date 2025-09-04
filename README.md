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
-   `examples/`: Contains an example script to train the MA-POCA agent on a PettingZoo environment.
-   `tests/`: Contains tests for the project.

## Development

### Setup

To set up the development environment, install the project in editable mode. This will also install all the required dependencies from `pyproject.toml`.

```bash
pip install -e .
```

To install development dependencies (like `pytest`), run:
```bash
pip install -e .[dev]
```

### Usage

To run the example training script for MA-POCA on the `simple_spread_v3` environment:

```bash
python -m examples.train_mpe
```

This will start the training process. Metrics and results will be logged to the console and also saved to an `mlruns` directory, which can be viewed with the MLflow UI:

```bash
mlflow ui
```

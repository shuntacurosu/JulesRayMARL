# Custom Reinforcement Learning Agent

This project is a customized distributed reinforcement learning agent built using [Ray](https://www.ray.io/). It is designed to implement various algorithms and distributed architectures, with the ultimate goal of implementing advanced multi-agent algorithms like Agent57.

## Core Technologies

-   [Ray](https://www.ray.io/): For distributed computing.
-   [MLFlow](https://mlflow.org/): For experiment tracking.
-   [Gymnasium](https://gymnasium.farama.org/): As the environment API.
-   [PettingZoo](https://pettingzoo.farama.org/): For multi-agent environments.

## Project Structure

-   `.github/`: Contains GitHub Actions workflows and templates.
-   `docs/`: Contains project documentation (specifications, design docs).
-   `src/`: Contains the source code for the RL agent.
-   `tests/`: Contains tests for the project.

## Development

This project follows a Test-Driven Development (TDD) approach. All code changes must be accompanied by corresponding tests.

### Setup

To set up the development environment, install the project in editable mode with the development dependencies:

```bash
pip install -e .[dev]
```

### Running Tests

To run the test suite:

```bash
pytest
```

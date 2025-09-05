# Architecture Design

This document describes the high-level architecture of the MA-POCA implementation for RLlib.

## Overview

The system is designed as a distributed application using the Ray framework with the following key components:

![Architecture Diagram](placeholder.png)

## Components

-   **Agent**: Implements the MA-POCA algorithm with a self-attention based centralized critic.
-   **Environment**: Uses PettingZoo multi-agent environments, specifically the simple spread environment.
-   **Trainer**: Built on Ray RLlib's PPO implementation, customized for MA-POCA.
-   **Experiment Tracking**: Uses MLflow to track training metrics and results.

## Data Flow

1. The environment generates observations for each agent.
2. Agents process observations through their policy networks to determine actions.
3. Actions are executed in the environment, producing rewards and new observations.
4. Experience tuples (observations, actions, rewards, next observations) are collected.
5. The trainer uses these experiences to update the policy networks.
6. Training metrics are logged to MLflow for analysis and visualization.
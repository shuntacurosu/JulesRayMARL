---
trigger: always_on
alwaysApply: true
---

# MARL Development Rules

## 1. Promoting TDD Development
- All feature development must be done using Test-Driven Development (TDD)
- Create test cases before implementing new features
- Implement tests using [pytest](file:///d:/Workspace/JulesRayMARL/.venv/Lib/site-packages/pytest/__init__.py)
- Regularly check test coverage to ensure sufficient comprehensiveness

## 2. Active Use of Existing Libraries
- When implementing new features, first consider whether they can be achieved with existing libraries or frameworks
- Minimize reinventing the wheel and actively use existing reliable libraries
- Especially for reinforcement learning related features, maximize the use of existing libraries such as [Ray RLlib](file:///d:/Workspace/JulesRayMARL/.venv/Lib/site-packages/ray/rllib/__init__.py) and [PettingZoo](file:///d:/Workspace/JulesRayMARL/.venv/Lib/site-packages/pettingzoo/__init__.py)

## 3. Active Use of MCP
- When questions or problems arise during development, actively use MCP to solve them
- Position MCP as an important tool for improving development efficiency and actively utilize it
- All thinking must be in English, and all output characters must be in Japanese

## 4. Implementation of Ruff Checks and Corrections
- Always perform code quality checks using Ruff before committing
- Detect errors with `ruff check` and format code with `ruff format`
- Fix all Ruff errors before committing

## 5. Keeping the docs Folder Up-to-Date
- Always keep the [docs](file:///d:/Workspace/JulesRayMARL/docs) folder up-to-date before committing
- Update related documentation when new features or changes occur
- Documentation is important for understanding and maintainability of the project, so always keep it up-to-date

## 6. Branch Management
- Direct pushing to the main branch is prohibited
- Always create a new branch for development

## 7. Development Environment
- The development environment is Windows
- The terminal is Windows PowerShell
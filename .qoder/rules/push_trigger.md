---
trigger: pre_commit
---

# Push Trigger Rules

## 1. Documentation Update
- Always keep the [docs](file:///d:/Workspace/JulesRayMARL/docs) folder up-to-date before committing
- Update related documentation when new features or changes occur

## 2. Ruff Checks and Corrections
- Always perform code quality checks using Ruff before committing
- Detect errors with `ruff check` and format code with `ruff format`
- Fix all Ruff errors before committing

## 3. Merge Request Management
- Check if a Merge Request (MR) has been created
- If no MR exists:
  - Create a new MR
  - Set the MR to automatically delete the branch when closed
- If an MR exists:
  - Push changes to the corresponding branch
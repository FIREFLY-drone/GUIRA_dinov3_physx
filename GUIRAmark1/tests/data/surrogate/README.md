# Surrogate Model Test Data

This directory contains minimal sample data for testing the FireSpreadNet surrogate model.

## Contents

- Sample PhysX simulation outputs (synthetic)
- Minimal dataset for unit testing
- Test fixtures for model training and inference

## Generating Test Data

Run the following to generate minimal test data:

```bash
cd integrations/guira_core/orchestrator/surrogate
python generate_ensemble.py --output-dir ../../../../tests/data/surrogate/test_dataset --n-runs 10 --n-timesteps 5
```

This creates a small dataset with 10 runs for testing purposes.

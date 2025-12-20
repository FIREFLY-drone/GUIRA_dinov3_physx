# GUIRA Core Integration

GUIRA core integration: DINOv3 & PhysX integration scaffolding.
See docs/DESIGN.md for full design.

## Overview

This module integrates NVIDIA PhysX fire simulations and Meta's DINOv3 self-supervised vision into the GUIRA wildfire intelligence platform.

## Structure

- `vision/` - DINOv3 vision services and detection probes
- `simulation/` - PhysX fire simulation components
- `orchestrator/` - Scheduling and surrogate model management
- `infra/` - Infrastructure templates and configurations
- `docs/` - Design documents and API schemas
- `samples/` - Sample data and test cases
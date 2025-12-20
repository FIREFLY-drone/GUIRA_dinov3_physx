"""
PhysX Fire Spread Omniverse Prototype (PH-06)

Rapid prototype for fire spread simulation using NVIDIA Omniverse PhysX.
"""

from pathlib import Path

__version__ = "0.1.0"
__all__ = ["run_simulation", "generate_geojson_output"]

# Import main functions if available
try:
    from .run_prototype import run_simulation, generate_geojson_output
except ImportError:
    # Allow module import even if dependencies not met
    pass

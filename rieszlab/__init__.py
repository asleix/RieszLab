"""RieszLab: A Python toolkit for monogenic analysis of spatio-temporal signals.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from .riesz import (
    MonogenicResult,
    RieszConfig,
    RieszResult,
    monogenic,
    orientation_to_slowness,
)
from .tensors import (
    GaussianTensorConfig,
    SteeredTensorConfig,
    TensorConfig,
    TensorResult,
    structure_tensor,
)
from .enhancement import (
    DirectionSelectionConfig,
    EnhancementResult,
    direction_weight,
    enhance_coherent,
    enhance_directional,
    reconstruct,
)

__version__ = "0.4.0"
__all__ = [
    "MonogenicResult",
    "RieszConfig",
    "RieszResult",
    "TensorConfig",
    "GaussianTensorConfig",
    "SteeredTensorConfig",
    "TensorResult",
    "monogenic",
    "orientation_to_slowness",
    "structure_tensor",
    "EnhancementResult",
    "enhance_coherent",
    "DirectionSelectionConfig",
    "direction_weight",
    "enhance_directional",
    "reconstruct",
]

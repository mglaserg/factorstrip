"""FactorStrip V2 research infrastructure.

The V2 package is intentionally separate from the exploratory V1 modules.  It
contains only research plumbing, factor/residual construction, and reference
implementations.  Inferential performance evaluation belongs behind the
EdgeLab preregistration gate.
"""

from .config import ResearchDesign, UniverseConfig
from .rw_r1000 import RwR1000Config
from .styles import StyleConfig

__all__ = ["ResearchDesign", "UniverseConfig", "RwR1000Config", "StyleConfig"]

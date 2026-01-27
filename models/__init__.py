from .backbones import __all__
from .bbox import __all__
from .hook import __all__
from .model_utils import __all__

# NOTE: necks may depend on optional CUDA extensions (e.g., bev_pool_v2_ext).
# Keep import soft to allow lightweight unit tests (e.g., GGF) to run when
# extensions are not built.
try:
    from .necks import __all__  # noqa: F401
except Exception:
    # Avoid hard failure when optional extensions are unavailable.
    pass

from .racformer import RaCFormer
from .racformer_head import RaCFormer_head
from .racformer_transformer import RaCFormerTransformer
from .rwhi import RWHIModule, AlphaMLP, AlphaEncoder, build_rwhi
from .ggf import (
    GGFModule, 
    GeometryFieldBuilder, 
    NativeRGF, 
    MGCModule, 
    GGAModule, 
    build_ggf,
    GGFDebugger,
)

__all__ = [
    'RaCFormer', 'RaCFormer_head', 'RaCFormerTransformer',
    'RWHIModule', 'AlphaMLP', 'AlphaEncoder', 'build_rwhi',
    # GGF2.0 模块
    'GGFModule', 'GeometryFieldBuilder', 'NativeRGF', 
    'MGCModule', 'GGAModule', 'build_ggf', 'GGFDebugger',
]

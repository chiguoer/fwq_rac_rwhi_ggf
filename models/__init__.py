from .backbones import __all__
from .bbox import __all__
from .hook import __all__
from .model_utils import __all__

# NOTE: necks may depend on optional CUDA extensions (e.g., bev_pool_v2_ext).
# Keep import soft to allow lightweight unit tests (e.g., GGF) to run when
# extensions are not built, but avoid swallowing unrelated errors.
try:
    from .necks import __all__  # noqa: F401
except Exception as exc:
    if 'bev_pool_v2_ext' in str(exc):
        import warnings
        warnings.warn(
            'Optional CUDA extension bev_pool_v2_ext is not available; '
            'necks will not be registered for this session.'
        )
    else:
        raise

from .racformer import RaCFormer
from .racformer_head import RaCFormer_head
from .racformer_transformer import RaCFormerTransformer
from .rwhi import RWHIModule, AlphaMLP,  build_rwhi
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
    'RWHIModule', 'AlphaMLP',  'build_rwhi',
    # GGF2.0 模块
    'GGFModule', 'GeometryFieldBuilder', 'NativeRGF', 
    'MGCModule', 'GGAModule', 'build_ggf', 'GGFDebugger',
]

from .backbones import __all__
from .bbox import __all__
from .necks import __all__
from .hook import __all__
from .model_utils import __all__

from .racformer import RaCFormer
from .racformer_head import RaCFormer_head
from .racformer_transformer import RaCFormerTransformer

# 新增模块导入 - 来自HGSFusion和RCBEVDet的融合模块
from .rhgm import RHGM, RHGMWrapper
from .radar_bev_net import RadarBEVNet, RadarBEVNetWrapper

# RWHI模块 - RCS加权混合锚点初始化
from .rwhi import RWHIModule, RWHIQueryGenerator

__all__ = [
    'RaCFormer', 'RaCFormer_head', 'RaCFormerTransformer',
    # 融合模块
    'RHGM', 'RHGMWrapper', 'RadarBEVNet', 'RadarBEVNetWrapper',
    # RWHI模块
    'RWHIModule', 'RWHIQueryGenerator'
]

# ============================================================
# RaCFormer 基线配置文件 (Baseline Configuration)
# ============================================================
# 用于消融实验：纯原始 RaCFormer，不使用 RHGM 和 RadarBEVNet
# 
# 使用方法：
#   训练：python train.py --config configs/racformer_baseline.py
#   测试：python val.py --config configs/racformer_baseline.py --weights checkpoints/xxx.pth
#
# 与增强版的区别：
#   - use_rhgm = False (关闭RHGM雷达点云增强)
#   - use_radar_bev_net = False (使用原始PillarFeatureNet编码器)
# ============================================================

# 导入原始配置作为基础
_base_ = './racformer_r50_nuimg_704x256_f8.py'

# ============================================================
# 显式设置：禁用所有增强模块，使用原始 RaCFormer
# ============================================================
# 注意：原始配置中没有这些参数，模型代码默认 use_rhgm=False 和 use_radar_bev_net=False
# 这里显式设置以便清晰表明是基线配置

# 如果需要显式覆盖（在使用 _base_ 继承时）：
# model = dict(
#     use_rhgm=False,
#     use_radar_bev_net=False,
# )

# ============================================================
# 消融实验说明
# ============================================================
# 
# 基线 (Baseline):
#   - 使用此配置文件
#   - 雷达编码器: PillarFeatureNet -> PointPillarsScatter
#   
# +RHGM:
#   - 修改 model.use_rhgm = True
#   - 添加 model.rhgm_cfg = {...}
#
# +RadarBEVNet:
#   - 修改 model.use_radar_bev_net = True
#   - 添加 model.radar_bev_net_cfg = {...}
#
# +RHGM+RadarBEVNet:
#   - 使用 configs/racformer_with_rhgm_radarbevnet.py
# ============================================================


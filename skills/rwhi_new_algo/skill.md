请你用中文回复我

每次你读到这个文件就请进入skills模式

进入skills模式是先执行并解决当前chat输出的指令，当操作者说“请执行固定步骤之后，在开始执行下面的‘强制性工作流程’”

1. 强制性工作流程
步骤 1：自审（/review 模式）
任何代码修改后立即触发 /review 模式。
代码修改自审：若是前面进行了代码修改任务，请先git --no-pager diff HEAD 出你修改的内容， 再去比对一下你的修改是否完成了指令所要求的内容。
审查重点范围：批归一化（BN）维度不匹配、梯度截断、静态形状违规、数值不稳定性等等。（审阅代码可以参考本文档最后的历次故障及其解决措施）

步骤 2：冒烟测试
环境：首先激活 conda 环境 "racrwhi"（执行命令：conda activate racrwhi）。
执行命令：运行与修改模块对应的配置文件（例如，针对 RWHI 相关修改：torchrun --nproc_per_node 2 train.py --config configs/racformer_with_rwhi.py）。
验证目标：确保前几轮训练迭代中无运行时错误（Runtime Error）：正式训练开始前，请仔细查看训练日志中的错误信息。若未发现错误，可跳过冒烟测试；若存在错误，请勿急于改写代码，需先整体浏览代码、理解代码的核心逻辑与原本意图后，再以 “最小修改原则” 修复错误代码 —— 仅修改引发错误的部分，不重写整个文件，也不改变代码原本的设计意图。若遇到棘手的问题无法自行解决，请先向我确认后再进行下一步操作。
本次会话的固定冒烟流程：conda activate racrwhi && python train.py --config configs/racformer_with_rwhi.py，至少跑满 10+ iterations 且不得出现梯度相关报错；若未跑通，禁止立刻改代码，需先输出修改计划待确认；若跑通，再继续做逻辑/性能检查。

步骤 3：代码提交（Git Push）
若未发现任何问题且训练正常进行，可将修改推送至 GitHub。
执行命令（按顺序）：git add .、git commit -m "[清晰的英文提交信息，例如：fix: deterministic jitter for RWHI v5.2]"、git push。
注意：Git 环境已预先配置 SSH/Token；无需输入用户名 / 密码。


2. 故障排查（简化版）
注意每次审阅代码或者运行测试时，出现的问题和解决方案请概括后归纳入skill.md文件当中（如下格式，注意：仅在
出现故障时传入相关事件，作为之后的编程经验）：
现象	                        解决方案
ONNX 导出失败	                检查并移除动态算子（如动态形状、条件循环）。
BN 运行时错误	                将 nn.BatchNorm1d 替换为 nn.LayerNorm（避免稀疏批次崩溃）。
评估阶段 mAP 波动	            确保仅使用固定噪声（评估模式下禁用 torch.rand）。
MLP 中梯度消失	                验证 alpha / 锚点（anchors）无意外的 detach () 操作（保持 requires_grad=True）。
Atan2Backward0 NaN/梯度崩溃（sin/cos 为零或 NaN）	在 decode_bbox 中对 sin/cos 做 nan_to_num + near_zero 掩码强制为 (0,1)，再归一化后 atan2，避免 (0,0) 反传 NaN。
ClampBackward 版本冲突（_validate_query_bbox 就地 clamp）	改为 out-of-place clamp（分拆 theta/d/z clamp 后重新 concat），避免对同一张量的就地修改破坏计算图版本。
GGF 单测导入失败（bev_pool_v2_ext 缺失）	在 models/__init__.py 中对 necks 导入做 try/except，避免非必要 CUDA 扩展导致 GGF 单测中断。
NativeRGF 报错：'float' object has no attribute 'clamp'	use_velocity_anisotropy 分支中 sigma_y 保持 tensor（sigma_tangent = sigma_y），避免 float 覆盖。
冒烟训练失败：data/nuscenes/v1.0-trainval 不存在	补齐 nuScenes 数据集或修正 dataroot 后再跑训练。
Cholesky 分解失败（Sigma_uv 非正定）	Sigma_uv 由 MGC 投影得到，训练早期可能非 SPD；已通过对称化 + 特征值 clamp + SPD 投影 + fallback（对角缩放）解决；若新数据集仍报错可调 eps / max_val。
MGC debug 日志缺失（debug_mgc=True 仍无输出）	检查 GGF cached params 是否生成；确认 use_image_sampling=True；必要时将 MGC debug 改为 logging 或 tee stdout 保存。
DataLoader worker 启动报错：cannot pickle 'dict_keys' object	将 dataset.eval_detection_configs.class_names 从 dict_keys 转为 list（在 CustomNuScenesDataset __init__ 里 list(...)）。

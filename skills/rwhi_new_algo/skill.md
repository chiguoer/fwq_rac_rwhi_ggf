请你用中文回复我

1. 强制性工作流程
步骤 1：冒烟测试
环境：首先激活 conda 环境 "racrwhi"（执行命令：conda activate racrwhi）。
执行命令：运行与修改模块对应的配置文件（例如，针对 RWHI 相关修改：torchrun --nproc_per_node 2 train.py --config configs/racformer_with_rwhi.py）。
验证目标：确保前几轮训练迭代中无运行时错误（Runtime Error）：正式训练开始前，请仔细查看训练日志中的错误信息。若未发现错误，可跳过冒烟测试；若存在错误，请勿急于改写代码，需先整体浏览代码、理解代码的核心逻辑与原本意图后，再以 “最小修改原则” 修复错误代码 —— 仅修改引发错误的部分，不重写整个文件，也不改变代码原本的设计意图。若遇到棘手的问题无法自行解决，请先向我确认后再进行下一步操作。
本次会话的固定冒烟流程：conda activate racrwhi && python train.py --config configs/racformer_with_rwhi.py，至少跑满 10+ iterations 且不得出现梯度相关报错；若未跑通，禁止立刻改代码，需先输出修改计划待确认；若跑通，再继续做逻辑/性能检查。


要求：在进入步骤 2 前修复所有发现的问题。
步骤 2：自审（/review 模式）
任何代码修改后立即触发 /review 模式。
审查重点范围：批归一化（BN）维度不匹配、梯度截断、静态形状违规、数值不稳定性。（审阅代码可以参考本文档最后的历次故障及其解决措施）
一般性审阅角度：
审查维度	检查内容关键词	
接口一致性	config 参数传递、魔法数	
维度安全	shape 明确、维度注释、兼容多卡
坐标/单位规范	归一化一致性、反归一化风险	
数值稳定性	clamp、epsilon、log 爆炸	
梯度路径	detach、.data、requires_grad	
模块可拓展性	写死结构、函数耦合	
部署静态图友好性	torch.rand、if 判断、动态 shape
风格规范	命名、注释、模块划分	

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

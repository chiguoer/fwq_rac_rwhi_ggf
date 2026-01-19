# Skill: RWHI v3 新算法闭环

## Goal
- RWHI v3（uniform base + d^4 + α-MLP + scatter_add + maxpool + global topk）
- 输出 anchors 固定为 `(B, K, 3)`，不使用动态输出算子

## Inputs/Outputs
- 输入：
  - `radar_points`: `(B, M, 5)`，含 padding（无效区可为 0）
  - `mask`: `(B, M)`，1=有效点，0=padding
- 输出：
  - `anchors`: `(B, K, 3)`，`(theta, d, z_norm)`

## TRT/Static Graph Constraints
- 禁止 `nonzero/unique` 等动态输出算子
- 禁止 Python for-loop 遍历点
- `topk` 的 K 必须固定
- 必须走 `scatter_add` 注入路径

## Validation / Acceptance Criteria
- conda 环境 `racrwhi` 下 `python tools/smoke_test_rwhi.py` 通过
- 单元测试通过（`pytest -q` 或独立脚本）
- ONNX 导出成功，并通过 `onnx.checker` 校验
- run.sh 最终打印 `RWHI skill run PASS`

## Common Failure Modes & Fixes
- onnx 缺失/版本不匹配：确认 `pip show onnx`，必要时重装
- shape 不固定导致 export 失败：设置 `max_points`、固定 `num_query`
- mask 未乘导致 padding 污染 topk：在权重计算后强制乘 `radar_mask`
- scatter_add index 越界：检查 pc_range、grid_size 与 clamp 逻辑

## 工作流约束
- 仅分析与修改 `rwhi` 相关代码，避免改动 `rhgm` 和 `radar_bev_net`。
- 每次更改代码后，在 conda 环境 `racrwhi` 运行与本次修改模块对应的训练配置：
  - 若修改的是 `rwhi` 相关代码，使用 `configs/racformer_with_rwhi.py`。
  - 若修改的是其他模块，使用该模块对应生成/绑定的配置文件。
  - 用于验证是否有报错；若无报错且训练开始，立即终止该进程。

## 变更后检查流程
1. 自动展示本次代码变更差异（类似 `git diff`）。
2. 切换到代码审查模式，检查：
   - 变更是否解决了原有问题
   - 是否引入新的潜在问题
   - 代码逻辑是否清晰合理
   - 是否有遗漏的边界情况
3. 提供具体的改进建议或潜在风险提示。

## 运行错误分析流程
1. 详细分析问题产生的原因。
2. 归纳总结根本原因和解决方案。
3. 使用标准化格式整理分析结果：
   - 问题描述
   - 错误现象
   - 根本原因
   - 解决方案
   - 预防措施
4. 建议将这些经验记录到技能文档中，形成可复用知识库。

## 标准化记录模板
```
问题描述:
错误现象:
根本原因:
解决方案:
预防措施:
```

## 经验记录规范
- 记录位置：本文件新增“问题复盘”小节，按日期追加。
- 内容要求：简洁、可复用、包含触发条件与验证方式。

## 问题复盘
- 暂无，待补充。

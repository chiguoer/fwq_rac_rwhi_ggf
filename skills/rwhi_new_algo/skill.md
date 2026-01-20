**### 1. MANDATORY WORKFLOW (NON-NEGOTIABLE)**
Step 1: Self-Review (/review Mode)
- Trigger /review mode immediately after ANY code modification.
- Focus review scope: BN dimension mismatch, gradient truncation, static shape violation, numerical instability.
- Requirement: Fix all identified issues before proceeding to Step 2.

Step 2: Smoke Test
- Environment: Activate conda environment "racrwhi" first (conda activate racrwhi).
- Execution command: Run the config file corresponding to the modified module (e.g., for RWHI changes: torchrun --nproc_per_node 2 train.py --config configs/racformer_with_rwhi.py).
- Validation target: Ensure no Runtime Error in the first few training iterations (stop manually if no error; fix immediately if error occurs).

Step 3: Code Commit (Git Push)
- Execution commands (in order): git add ., git commit -m "[clear English commit message, e.g., fix: deterministic jitter for RWHI v5.2]", git push.
- Note: Git environment is pre-configured with SSH/Token; no need to enter username/password.
- Note: If no issues are found and training proceeds normally, the changes can be pushed to GitHub.

### 2. RWHI v5.2 Algorithm Specs
Core Theory: "Isotropic Perturbed Field" + "Score Injection".
Key Constraints (MUST COMPLY):
- Determinism: Forbid torch.rand in evaluation/inference (only allow fixed texture-based jitter).
- Output Range: Anchors must be normalized to [0, 1] space (strict alignment with RaCFormer baseline).
- Static Graph: No loops, no dynamic shape operations (compatible with TensorRT/ONNX export).

### 3. Troubleshooting (Simplified)
| Phenomenon                | Solution                                  |
|---------------------------|-------------------------------------------|
| ONNX export failure       | Check and remove dynamic operators (e.g., dynamic shape, conditional loops). |
| BN runtime error          | Replace nn.BatchNorm1d with nn.LayerNorm (avoids sparse batch crash). |
| mAP fluctuation in eval   | Ensure fixed noise only (disable torch.rand in eval mode). |
| Gradient vanishing in MLP | Verify no unintended detach() on alpha/anchors (keep requires_grad=True). |

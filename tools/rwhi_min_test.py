import math
import os
import sys
import types
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Avoid importing models/__init__.py to keep test lightweight.
models_pkg = types.ModuleType("models")
models_pkg.__path__ = [os.path.join(REPO_ROOT, "models")]
sys.modules["models"] = models_pkg

bbox_pkg = types.ModuleType("models.bbox")
bbox_pkg.__path__ = [os.path.join(REPO_ROOT, "models", "bbox")]
sys.modules["models.bbox"] = bbox_pkg

from models.rwhi import RWHIModule


def make_radar_points(num_near, num_far, device, seed=0):
    torch.manual_seed(seed)
    total = num_near + num_far
    if total == 0:
        return torch.zeros(1, 0, 5, device=device)

    # Near field points: 5-25m
    near_r = 5.0 + 20.0 * torch.rand(num_near, device=device)
    near_theta = 2.0 * math.pi * torch.rand(num_near, device=device)
    near_x = near_r * torch.cos(near_theta)
    near_y = near_r * torch.sin(near_theta)

    # Far field points: 35-50m
    far_r = 35.0 + 15.0 * torch.rand(num_far, device=device)
    far_theta = 2.0 * math.pi * torch.rand(num_far, device=device)
    far_x = far_r * torch.cos(far_theta)
    far_y = far_r * torch.sin(far_theta)

    x = torch.cat([near_x, far_x], dim=0)
    y = torch.cat([near_y, far_y], dim=0)
    z = torch.zeros_like(x)

    # RCS with large dynamic range; far points generally stronger.
    rcs_near = 0.1 + 2.0 * torch.rand(num_near, device=device)
    rcs_far = 5.0 + 40.0 * torch.rand(num_far, device=device)
    rcs = torch.cat([rcs_near, rcs_far], dim=0)

    # Radial velocity
    v_r = -5.0 + 10.0 * torch.rand(total, device=device)

    points = torch.stack([x, y, z, rcs, v_r], dim=-1)
    return points.unsqueeze(0)


def radius_from_theta_d(theta_d, polar_radius):
    d = theta_d[..., 1]
    return d * polar_radius


def check_case(case_name, module, radar_points):
    module.eval()
    with torch.no_grad():
        out1, mask1 = module(radar_points)
        out2, _ = module(radar_points)

    num_query = module.num_query
    num_safety = module.num_safety
    polar_radius = module.polar_radius

    print(f"\n[{case_name}] query_bbox shape: {tuple(out1.shape)}")
    print(f"[{case_name}] anchor_mask shape: {tuple(mask1.shape)}")
    assert list(out1.shape) == [radar_points.shape[0], num_query, 10]
    assert list(mask1.shape) == [radar_points.shape[0], num_query]
    assert not mask1[:, :num_safety].any()
    assert mask1[:, num_safety:].all()

    safety = out1[:, :num_safety, :]
    saliency = out1[:, num_safety:, :]

    safety_radius = radius_from_theta_d(safety[..., :2], polar_radius)
    safety_max = safety_radius.max().item() if safety_radius.numel() > 0 else 0.0
    print(f"[{case_name}] safety radius max (m): {safety_max:.3f}")
    assert safety_max <= module.safety_max_range + 1e-3

    saliency_radius = radius_from_theta_d(saliency[..., :2], polar_radius)
    valid = saliency_radius > 1e-3
    if valid.any():
        saliency_min = saliency_radius[valid].min().item()
        print(f"[{case_name}] saliency radius min (m): {saliency_min:.3f}")
        assert saliency_min > module.safety_max_range
    else:
        print(f"[{case_name}] saliency anchors empty after mask/fallback")

    diff = (out1 - out2).abs().max().item()
    print(f"[{case_name}] eval determinism max diff: {diff:.6f}")
    assert diff == 0.0


def main():
    device = torch.device("cpu")
    module = RWHIModule(
        num_query=1200,
        num_safety=900,
        num_saliency=300,
    ).to(device)

    print(f"[RWHI] map_size={module.map_size:.3f}, polar_radius={module.polar_radius:.3f}")

    # Case 1: mixed near + far points
    radar_mixed = make_radar_points(num_near=200, num_far=200, device=device, seed=1)
    check_case("mixed", module, radar_mixed)

    # Case 2: only near points (should be masked out by saliency)
    radar_near = make_radar_points(num_near=300, num_far=0, device=device, seed=2)
    check_case("near_only", module, radar_near)

    # Case 3: empty point cloud (deterministic fallback)
    radar_empty = make_radar_points(num_near=0, num_far=0, device=device, seed=3)
    check_case("empty", module, radar_empty)

    print("\n[RWHI] min test completed")


if __name__ == "__main__":
    main()

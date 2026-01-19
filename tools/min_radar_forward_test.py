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

from models.radar_bev_net import RadarBEVNet


def main():
    torch.manual_seed(0)
    model = RadarBEVNet(
        in_channels=7,
        feat_channels=(8,),
        voxel_size=(0.5, 0.5, 1.0),
        point_cloud_range=(-5.0, -5.0, -1.0, 5.0, 5.0, 1.0),
    )
    model.train()

    n_voxels = 4
    max_points = 6
    features = torch.zeros(n_voxels, max_points, 7)
    features[:, :, 0] = torch.empty(n_voxels, max_points).uniform_(-4.5, 4.5)
    features[:, :, 1] = torch.empty(n_voxels, max_points).uniform_(-4.5, 4.5)
    features[:, :, 2] = torch.empty(n_voxels, max_points).uniform_(-0.5, 0.5)
    features[:, :, 3:] = torch.randn(n_voxels, max_points, 4)

    num_voxels = torch.full((n_voxels,), max_points, dtype=torch.long)
    coors = torch.zeros(n_voxels, 3, dtype=torch.long)
    coors[:, 1] = torch.arange(n_voxels) % 2
    coors[:, 2] = torch.arange(n_voxels) // 2

    out = model(features, num_voxels, coors)
    loss = out.mean()
    loss.backward()

    print(f"[RadarBEVNet] output shape: {tuple(out.shape)}")
    print(f"[RadarBEVNet] loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()

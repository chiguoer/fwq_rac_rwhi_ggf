import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ggf import MGCModule, GGAModule
from models.bbox.utils import xy2theta_d_coods


def _build_dummy_img_metas(image_h=100, image_w=100, device='cpu'):
    # 简单 pinhole 投影：u = fx*x/z + cx, v = fy*y/z + cy
    fx = fy = 50.0
    cx = image_w / 2.0
    cy = image_h / 2.0
    lidar2img = torch.tensor([
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], device=device).view(1, 1, 4, 4)

    img_metas = [{
        'lidar2img': lidar2img,
        'img_shape': [(image_h, image_w, 3)],
    }]
    return img_metas


def test_mgc_sampling_center():
    device = 'cpu'
    mgc = MGCModule(
        use_image_sampling=True,
        sample_res=(3, 4),
        debug_mgc=False,
    ).to(device)

    # Query at center (x=0,y=0), z mid
    xy = torch.tensor([[[0.5, 0.5]]], device=device)
    theta_d = xy2theta_d_coods(xy)
    query_bbox = torch.zeros(1, 1, 10, device=device)
    query_bbox[..., 0:2] = theta_d[..., 0:2]
    query_bbox[..., 2:3] = 0.5  # normalized z

    query_feat = torch.zeros(1, 1, 256, device=device)

    gaussian_params = {
        'centers': torch.tensor([[[0.0, 0.0]]], device=device),
        'sigmas': torch.tensor([[[1.0, 1.0]]], device=device),
        'theta': torch.tensor([[0.0]], device=device),
        'mask': torch.tensor([[True]], device=device),
    }

    img_metas = _build_dummy_img_metas(device=device)
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    sampling_locations, info = mgc.build_image_sampling_locations(
        query_bbox, query_feat, gaussian_params, img_metas, pc_range, sample_res=(3, 4)
    )
    assert sampling_locations is not None
    grid_xy = sampling_locations[..., :2]
    center = grid_xy.mean(dim=2)
    # 预期中心接近 (0.5, 0.5)
    assert torch.allclose(center, torch.tensor([[[0.5, 0.5]]]), atol=0.1)


def test_gga_bias_order():
    device = 'cpu'
    gga = GGAModule(num_heads=1, learnable_temperature=False).to(device)

    query_bbox = torch.zeros(1, 1, 10, device=device)
    query_bbox[..., 0:1] = 0.0
    query_bbox[..., 1:2] = 0.0
    query_bbox[..., 2:3] = 0.5
    query_feat = torch.zeros(1, 1, 256, device=device)

    gaussian_params = {
        'centers': torch.tensor([[[0.0, 0.0], [5.0, 0.0]]], device=device),
        'sigmas': torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], device=device),
        'mask': torch.tensor([[True, True]], device=device),
    }
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    bias = gga.compute_geometry_bias(query_bbox, query_feat, gaussian_params, pc_range)
    assert bias is not None
    # 更近的 Key（索引0）应有更大的 bias（更接近0）
    assert (bias[0, 0, 0, 0] > bias[0, 0, 0, 1]).item()


if __name__ == '__main__':
    test_mgc_sampling_center()
    test_gga_bias_order()
    print('GGF sanity checks passed.')

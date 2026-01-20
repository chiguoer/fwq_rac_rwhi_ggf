# -*- coding: utf-8 -*-
import os
import sys

import torch
import torch.nn as nn
import onnx

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.rwhi import RWHIModule


class RWHIExportWrapper(nn.Module):
    def __init__(self, rwhi):
        super().__init__()
        self.rwhi = rwhi

    def forward(self, radar_points, radar_mask):
        anchors, _ = self.rwhi(radar_points=radar_points, radar_mask=radar_mask)
        return anchors


def _shape_str(value_info):
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(str(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append('?')
    return '(' + ','.join(dims) + ')'


def main():
    device = torch.device('cpu')
    B, M, K = 1, 128, 64

    rwhi = RWHIModule(
        rwhi_version='v3',
        num_query=K,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=64,
        d_ref=30.0,
        beta=0.0,
        base_bias=1.0,
        z_default=0.5,
        max_points=M,
        radar_channel_map=dict(x=0, y=1, z=2, rcs=3, v_r=4),
        enabled=True,
    ).to(device)
    rwhi.eval()

    model = RWHIExportWrapper(rwhi).to(device)
    model.eval()

    radar_points = torch.randn(B, M, 5, device=device)
    radar_mask = torch.ones(B, M, device=device, dtype=torch.float32)

    out_dir = os.path.join(ROOT, 'artifacts')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'rwhi_v3.onnx')

    torch.onnx.export(
        model,
        (radar_points, radar_mask),
        out_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['radar_points', 'radar_mask'],
        output_names=['anchors'],
    )

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    print('ONNX export OK:', out_path)
    for inp in onnx_model.graph.input:
        print('input:', inp.name, _shape_str(inp))
    for out in onnx_model.graph.output:
        print('output:', out.name, _shape_str(out))


if __name__ == '__main__':
    main()

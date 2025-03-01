import torch
from torch import Tensor
import torch_scatter
from .kernel_manager import KernelManager
import time


try:
    import cupy as cp
    import pytorch_pfn_extras as ppe
    from torch.utils.dlpack import to_dlpack, from_dlpack
except:
    pass
import numpy as np

def _to_cupy(x):
    if x is not None:
        return cp.from_dlpack(to_dlpack(x))
    return 0


class ComputeLatticeFrameCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, a_ik, rpos_ij_e, dist2_min_e, tvecs_n, batch_i, edge_ij_e, K, dist_max, wscale, rvlen_n=None, cutoff_radius=None):
        N, H = a_ik.shape
        E = edge_ij_e.shape[1]
        kw = {'device': a_ik.device, 'dtype': a_ik.dtype}

        a_ik = a_ik.contiguous().detach()
        rpos_ij_e = rpos_ij_e.contiguous()
        tvecs_n = tvecs_n.contiguous()
        batch_i = batch_i.contiguous()
        dist2_min_e = dist2_min_e.contiguous() if dist2_min_e is not None else None
        edge_ij_e = edge_ij_e.contiguous()

        bsz = H
        dev = a_ik.device

        seed = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, dtype=np.int64).item()
        state_buff = torch.empty(E * H * 48, dtype=torch.uint8, device=dev)

        dx_first = torch.empty(N, H, **kw)
        dy_first = torch.empty(N, H, **kw)
        dz_first = torch.empty(N, H, **kw)

        dx_second = torch.empty(N, H, **kw)
        dy_second = torch.empty(N, H, **kw)
        dz_second = torch.empty(N, H, **kw)

        dx_third = torch.empty(N, H, **kw)
        dy_third = torch.empty(N, H, **kw)
        dz_third = torch.empty(N, H, **kw)

        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            from .. import global_config as config

            kernel = KernelManager.compute_lattice_first_frame
            kernel((E,), (H,), (
                _to_cupy(a_ik),
                _to_cupy(rpos_ij_e),
                _to_cupy(dist2_min_e),
                _to_cupy(tvecs_n),
                _to_cupy(batch_i),
                _to_cupy(edge_ij_e),
                N, H, E,
                K, dist_max, wscale,
                _to_cupy(rvlen_n), cutoff_radius,
                _to_cupy(state_buff),
                seed,
                _to_cupy(dx_first),
                _to_cupy(dy_first),
                _to_cupy(dz_first)
            ))

        seed = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, dtype=np.int64).item()
        state_buff = torch.empty(E * H * 48, dtype=torch.uint8, device=dev)

        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            from .. import global_config as config
            kernel = KernelManager.compute_lattice_second_frame
            kernel((E,), (H,), (
                _to_cupy(a_ik),
                _to_cupy(rpos_ij_e),
                _to_cupy(dist2_min_e),
                _to_cupy(tvecs_n),
                _to_cupy(batch_i),
                _to_cupy(edge_ij_e),
                N, H, E,
                K, dist_max, wscale,
                _to_cupy(rvlen_n), cutoff_radius,
                _to_cupy(state_buff),
                seed,
                _to_cupy(dx_first),
                _to_cupy(dy_first),
                _to_cupy(dz_first),
                _to_cupy(dx_second),
                _to_cupy(dy_second),
                _to_cupy(dz_second)
            ))

        seed = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, dtype=np.int64).item()
        state_buff = torch.empty(E * H * 48, dtype=torch.uint8, device=dev)

        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            from .. import global_config as config
            kernel = KernelManager.compute_lattice_third_frame
            kernel((E,), (H,), (
                _to_cupy(a_ik),
                _to_cupy(rpos_ij_e),
                _to_cupy(dist2_min_e),
                _to_cupy(tvecs_n),
                _to_cupy(batch_i),
                _to_cupy(edge_ij_e),
                N, H, E,
                K, dist_max, wscale,
                _to_cupy(rvlen_n), cutoff_radius,
                _to_cupy(state_buff),
                seed,
                _to_cupy(dx_first),
                _to_cupy(dy_first),
                _to_cupy(dz_first),
                _to_cupy(dx_second),
                _to_cupy(dy_second),
                _to_cupy(dz_second),
                _to_cupy(dx_third),
                _to_cupy(dy_third),
                _to_cupy(dz_third)
            ))

        frame_first = torch.empty((N, H, 3), **kw)
        frame_second = torch.empty((N, H, 3), **kw)
        frame_third = torch.empty((N, H, 3), **kw)

        frame_first[:, :, 0] = dx_first
        frame_first[:, :, 1] = dy_first
        frame_first[:, :, 2] = dz_first

        frame_second[:, :, 0] = dx_second
        frame_second[:, :, 1] = dy_second
        frame_second[:, :, 2] = dz_second

        frame_third[:, :, 0] = dx_third
        frame_third[:, :, 1] = dy_third
        frame_third[:, :, 2] = dz_third

        return frame_first, frame_second, frame_third

    @staticmethod
    def backward(ctx, gframe_first, gframe_second, gframe_third):
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None


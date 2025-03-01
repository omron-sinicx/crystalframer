import torch
from torch import Tensor
import torch_scatter
from .kernel_manager import KernelManager
import time
from .cupy_utils import *

try:
    import cupy as cp
    import pytorch_pfn_extras as ppe
    from torch.utils.dlpack import to_dlpack, from_dlpack
except:
    pass

from cupy.random import XORWOW
import numpy as np

def _to_cupy(x):
    if x is not None:
        return cp.from_dlpack(to_dlpack(x))
    return 0


class ComputeFrame1MaximumStaticMethodCUDA(torch.autograd.Function):
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
        bsz = 32
        dev = a_ik.device

        seed = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, dtype=np.int64).item()
        state_buff = torch.empty(E*H * 48, dtype=torch.uint8, device=dev)

        dx_max = torch.empty(E, H, **kw)
        dy_max = torch.empty(E, H, **kw)
        dz_max = torch.empty(E, H, **kw)
        maximum_value = torch.ones((E, H), **kw) * 1e10

        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            from .. import global_config as config

            kernel = KernelManager.compute_maximum_frame_each_edge
            kernel(((E * H + bsz - 1) // bsz,), (bsz,), (
                _to_cupy(a_ik),
                _to_cupy(rpos_ij_e),
                _to_cupy(dist2_min_e),
                _to_cupy(tvecs_n),
                _to_cupy(batch_i),
                _to_cupy(edge_ij_e),
                to_uint32(N), to_uint32(H), to_uint32(E),
                _to_cupy(rvlen_n), to_float32(cutoff_radius),
                _to_cupy(state_buff),
                seed,
                _to_cupy(dx_max),
                _to_cupy(dy_max),
                _to_cupy(dz_max),
                _to_cupy(maximum_value)
            ))

        out, argmax = torch_scatter.scatter_max(
            maximum_value, edge_ij_e[0, :].repeat(H, 1).T,
            out=torch.ones((N, H), **kw) * (-1e10), dim=0)
        index = torch.arange(H, device=a_ik.device).repeat(N, 1)

        frame_first = torch.empty((N, H, 3), **kw)
        frame_first[:, :, 0] = dx_max[argmax, index]
        frame_first[:, :, 1] = dy_max[argmax, index]
        frame_first[:, :, 2] = dz_max[argmax, index]

        return frame_first

    @staticmethod
    def backward(ctx, gframevec):
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None



class ComputeFrame2MaximumStaticMethodCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, a_ik, rpos_ij_e, dist2_min_e, tvecs_n, batch_i, edge_ij_e, K, dist_max, wscale, frame1, rvlen_n=None, cutoff_radius=None):
        a_ik = a_ik.to(torch.float32)
        q = q.to(torch.float32)
        k = k.to(torch.float32)

        N, H = a_ik.shape
        E = edge_ij_e.shape[1]
        kw = {'device': a_ik.device, 'dtype': a_ik.dtype}


        dx_first = frame1[:, :, 0].contiguous()
        dy_first = frame1[:, :, 1].contiguous()
        dz_first = frame1[:, :, 2].contiguous()

        dev = a_ik.device

        bsz = H
        bsz = 32

        dx_second = torch.empty((E, H), **kw)
        dy_second = torch.empty((E, H), **kw)
        dz_second = torch.empty((E, H), **kw)
        second_value = torch.empty((E, H), **kw)

        seed = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, dtype=np.int64).item()
        state_buff = torch.empty(E * H * 48, dtype=torch.uint8, device=dev)

        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            kernel = KernelManager.compute_second_frame_static_each_edge
            kernel(((E * H + bsz - 1) // bsz,), (bsz,), (
                    _to_cupy(a_ik),
                    _to_cupy(rpos_ij_e),
                    _to_cupy(dist2_min_e),
                    _to_cupy(tvecs_n),
                    _to_cupy(batch_i),
                    _to_cupy(edge_ij_e),
                    to_uint32(N), to_uint32(H), to_uint32(E),
                    _to_cupy(rvlen_n), to_float32(cutoff_radius),
                    _to_cupy(state_buff),
                    seed,
                    _to_cupy(dx_first),
                    _to_cupy(dy_first),
                    _to_cupy(dz_first),
                    _to_cupy(dx_second),
                    _to_cupy(dy_second),
                    _to_cupy(dz_second),
                    _to_cupy(second_value)
            ))

        out, argmax = torch_scatter.scatter_max(second_value, edge_ij_e[0, :].repeat(H, 1).T, out=torch.ones((N, H), **kw)*(-1e10), dim = 0)

        index = torch.arange(H, device = a_ik.device).repeat(N, 1)

        dx_second_max = dx_second[argmax, index]
        dy_second_max = dy_second[argmax, index]
        dz_second_max = dz_second[argmax, index]

        frame_second = torch.empty((N, H, 3), **kw)
        frame_second[:, :, 0] = dx_second_max
        frame_second[:, :, 1] = dy_second_max
        frame_second[:, :, 2] = dz_second_max


        return frame_second

    @staticmethod
    def backward(ctx, gframevec):
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


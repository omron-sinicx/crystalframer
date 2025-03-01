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


def _to_cupy(x):
    if x is not None:
        return cp.from_dlpack(to_dlpack(x))
    return 0


class ComputePCAFrame(torch.autograd.Function):
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

        mean_pos = torch_scatter.scatter_mean(rpos_ij_e, edge_ij_e[0, :], dim = 0)[edge_ij_e[0, :]]
        rpos_ij_e_subtract_mean = rpos_ij_e - mean_pos


        sigma_mat = rpos_ij_e_subtract_mean.reshape(E, -1, 1) @ rpos_ij_e_subtract_mean.reshape(E, 1, -1)
        sigma_mat_fused = torch_scatter.scatter_add(sigma_mat, edge_ij_e[0, :], dim = 0)

        eig_val, eig_vec = torch.linalg.eigh(sigma_mat_fused)

        frame_vec = eig_vec.reshape(N, 1, 3, 3)
        frame_vec = frame_vec.repeat(1, H, 1, 1)

        random_variable = torch.tensor([-1, 1], device = dev)
        idx = torch.randint(2, size=(N, H, 1), device = dev)
        random_variable_first = random_variable[idx].repeat(1, 1, 3)
        idx = torch.randint(2, size=(N, H, 1), device=dev)
        random_variable_second = random_variable[idx].repeat(1, 1, 3)
        idx = torch.randint(2, size=(N, H, 1), device=dev)
        random_variable_third = random_variable[idx].repeat(1, 1, 3)

        frame_first = frame_vec[:, :, :, 0] * random_variable_first
        frame_second = frame_vec[:, :, :, 1] * random_variable_second
        frame_third = frame_vec[:, :, :, 2] * random_variable_third

        frame_vec = torch.stack([frame_first, frame_second, frame_third], dim=-1)
        detval = torch.linalg.det(frame_vec).unsqueeze(-1).unsqueeze(-1)
        frame_vec = frame_vec * detval

        frame_first = frame_vec[:, :, :, 0]
        frame_second = frame_vec[:, :, :, 1]
        frame_third = frame_vec[:, :, :, 2]

        frame_first = frame_first.contiguous()
        frame_second = frame_second.contiguous()
        frame_third = frame_third.contiguous()

        return frame_first, frame_second, frame_third

    @staticmethod
    def backward(ctx, gframe_first, gframe_second, gframe_third):
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None


import torch
from torch import Tensor
from .kernel_manager import KernelManager

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

import numpy as np


class RealPeriodicEncodingWithProjFuncCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_ik, rpos_ij_e, dist2_min_e, tvecs_n, batch_i, edge_ij_e, K, dist_max, wscale, \
                W_k, frame_method, rvlen_n=None, cutoff_radius=None):

        # a_ik      : (points, heads)
        # rpos_ij_e : (edges, 3)
        # tvecs_n   : (batch, 3, 3)
        # batch_i   : (points)
        # edge_ij_e : (2, edges)
        # z_ijk = log( sum_n exp( a_ik*|pj + t1*n1+t2*n2+t3*n3 - pi|^2 ) )
        #           : (edges, heads)

        N, H = a_ik.shape
        E = edge_ij_e.shape[1]
        kw = {'device': a_ik.device, 'dtype': a_ik.dtype}

        a_ik = a_ik.contiguous().detach()
        rpos_ij_e = rpos_ij_e.contiguous()
        tvecs_n = tvecs_n.contiguous()
        batch_i = batch_i.contiguous()
        dist2_min_e = dist2_min_e.contiguous() if dist2_min_e is not None else None
        edge_ij_e = edge_ij_e.contiguous()
        if W_k is not None:
            W_k = W_k.detach().contiguous()
            assert W_k.dim() in (3, 4)
            W_num = 1 if W_k.dim() == 3 else W_k.shape[0]
            W_dim = W_k.shape[-2]
            v_ekd = torch.empty((E, H, W_dim), **kw) if K > 0 else None  # not neaded for noproj
        else:
            W_num = 0
            W_dim = 0
            v_ekd = torch.empty((E, H, K), **kw) if K > 0 else None
        z_ek = torch.empty((E, H), **kw)

        bsz = H
        dev = a_ik.device

        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            from .. import global_config as config
            kernel = KernelManager.real_enc_proj_fwd

            if frame_method == "no":
                sigma_mat = None
                sigma_mat_max = None
                kernel(((E * H + bsz - 1) // bsz,), (bsz,), (
                    _to_cupy(a_ik),
                    _to_cupy(rpos_ij_e),
                    _to_cupy(dist2_min_e),
                    _to_cupy(tvecs_n),
                    _to_cupy(batch_i),
                    _to_cupy(edge_ij_e),
                    N, H, E,
                    K, dist_max, wscale,
                    _to_cupy(W_k), W_num,
                    _to_cupy(rvlen_n), cutoff_radius,
                    _to_cupy(z_ek),
                    _to_cupy(v_ekd)
                ))
            else:
                sigma_mat = torch.zeros(E, H, 6, dtype=torch.float32, device=dev)
                edge_ij_e_select = torch.zeros((E, H), dtype=torch.int, device=dev)
                seed = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, dtype=np.int64).item()
                state_buff = torch.empty(E * H * 48, dtype=torch.uint8, device=dev)
                kernel = KernelManager.select_edges
                kernel((1,), (H,), (
                    _to_cupy(edge_ij_e),
                    _to_cupy(edge_ij_e_select),
                    _to_cupy(state_buff),
                    seed,
                    E, N, H
                ))
                seed = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, dtype=np.int64).item()
                state_buff = torch.empty(E * H * 48, dtype=torch.uint8, device=dev)
                sigma_mat_max = torch.zeros((N, H, 6), device = a_ik.device)

                kernel = KernelManager.real_enc_sigma_fwd
                kernel(((E * H + bsz - 1) // bsz,), (bsz,), (
                    _to_cupy(a_ik),
                    _to_cupy(rpos_ij_e),
                    _to_cupy(dist2_min_e),
                    _to_cupy(tvecs_n),
                    _to_cupy(batch_i),
                    _to_cupy(edge_ij_e),
                    N, H, E,
                    K, dist_max, wscale,
                    W_num,
                    _to_cupy(rvlen_n), cutoff_radius,
                    _to_cupy(sigma_mat),
                    _to_cupy(state_buff),
                    seed,
                    _to_cupy(edge_ij_e_select),
                    _to_cupy(sigma_mat_max)
                ))

        ctx.save_for_backward(a_ik, rpos_ij_e, dist2_min_e, tvecs_n, batch_i, edge_ij_e, rvlen_n, W_k, z_ek, v_ekd)
        ctx.K = K
        ctx.dist_max = dist_max
        ctx.wscale = wscale
        ctx.cutoff_radius = cutoff_radius
        ctx.frame_method = frame_method
        if K <= 0:
            return z_ek,

        return z_ek, v_ekd, sigma_mat, sigma_mat_max

    @staticmethod
    def backward(ctx, gz_ek, gv_ekd=None, gv_sigma_mat=None, sigma_mat_max = None):
        if ctx.frame_method == "no":
            # a_ik, rpos_ij_e, tvecs_n, batch_i, edge_ij_e, z_ek = ctx.saved_tensors[:6]
            a_ik, rpos_ij_e, dist2_min_e, tvecs_n, batch_i, edge_ij_e, rvlen_n, W_k, z_ek, v_ekd = ctx.saved_tensors
            K = ctx.K
            dist_max = ctx.dist_max
            wscale = ctx.wscale
            cutoff_radius = ctx.cutoff_radius
            N, H = a_ik.shape
            E = edge_ij_e.shape[1]

            e_start_i = torch.zeros(N + 1, dtype=batch_i.dtype, device=batch_i.device)
            e_start_i.scatter_add_(0, edge_ij_e[0] + 1, torch.ones_like(edge_ij_e[0]))
            e_start_i = e_start_i.cumsum(0)

            ga_ik = torch.empty_like(a_ik)

            dev = a_ik.device
            gW_k = None
            if W_k is not None:
                # W: (edges or 1, heads, head_dim, K)
                assert W_k.dim() in (3, 4)
                W_num = 1 if W_k.dim() == 3 else W_k.shape[0]
                W_dim = W_k.shape[-2]

                # W:     (edges or 1, heads, head_dim, K)
                # gv_ekd:(edges     , heads, Vdim)
                # v_ekd: (edges or 1, heads, K)
                gW_k = torch.empty((max(W_num, N),) + W_k.shape[-3:], device=dev, dtype=a_ik.dtype)
            else:
                W_num = 0
                W_dim = 0

            bsz = H
            with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
                if False and rvlen_n is None:
                    KernelManager.position_enc_proj_backward(((N * H + bsz - 1) // bsz,), (bsz,), (
                        _to_cupy(a_ik.detach()),
                        _to_cupy(rpos_ij_e),
                        _to_cupy(dist2_min_e),
                        _to_cupy(tvecs_n),
                        _to_cupy(batch_i),
                        _to_cupy(edge_ij_e),
                        _to_cupy(e_start_i),
                        _to_cupy(z_ek.detach()),
                        _to_cupy(gz_ek.detach().contiguous()),
                        _to_cupy(gv_ekd),
                        N, H, E,
                        K, dist_max, wscale,
                        _to_cupy(W_k), W_num,
                        _to_cupy(ga_ik),
                        _to_cupy(gW_k),
                    ))
                else:
                    from .. import global_config as config
                    kernel = KernelManager.real_enc_proj_bwd
                    kernel(((N * H + bsz - 1) // bsz,), (bsz,), (
                        _to_cupy(a_ik.detach()),
                        _to_cupy(rpos_ij_e),
                        # _to_cupy(dist2_min_e),
                        _to_cupy(tvecs_n),
                        _to_cupy(batch_i),
                        _to_cupy(edge_ij_e),
                        _to_cupy(e_start_i),
                        _to_cupy(z_ek.detach()),
                        _to_cupy(gz_ek.detach().contiguous()),
                        _to_cupy(gv_ekd),
                        N, H, E,  # K,
                        dist_max, wscale,
                        _to_cupy(W_k), W_num,
                        _to_cupy(rvlen_n), cutoff_radius,
                        _to_cupy(ga_ik),
                        _to_cupy(gW_k),
                    ))

            if rvlen_n is None:
                return ga_ik, None, None, None, None, None, None, None, None, gW_k, None, None, None

            return ga_ik, None, None, None, None, None, None, None, None, gW_k, None, None, None, None, None

        else:
            a_ik, rpos_ij_e, dist2_min_e, tvecs_n, batch_i, edge_ij_e, rvlen_n, W_k, z_ek, v_ekd = ctx.saved_tensors
            if rvlen_n is None:
                return None, None, None, None, None, None, None, None, None, None, None, None, None

            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

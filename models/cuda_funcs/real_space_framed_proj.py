import torch
from torch import Tensor
from .kernel_manager import KernelManager

try:
    import cupy as cp
    import pytorch_pfn_extras as ppe
    from torch.utils.dlpack import to_dlpack, from_dlpack
except:
    pass

from .cupy_utils import *

def _to_cupy(x):
    if x is not None:
        return cp.from_dlpack(to_dlpack(x))
    return 0

class RealEncodingWithFramedProjFuncCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_ik, rpos_ij_e, dist2_min_e, tvecs_n, batch_i, edge_ij_e, K, dist_max, wscale, \
                W_k, W_k1, W_k2, W_k3, frame_vec_1, frame_vec_2, frame_vec_3, dim_angle_enc, value_pe_angle_scale,
                cos_abs, length_rbf_mul, angle_rbf_mul, rvlen_n=None, cutoff_radius=None):

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
        dist_max = float(dist_max)

        if cos_abs == 0:
            angle_sigma = 2 * value_pe_angle_scale / (dim_angle_enc-1)
        else:
            angle_sigma = value_pe_angle_scale / (dim_angle_enc-1)

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

        bsz = KernelManager.PE_THREAD_NUM
        dev = a_ik.device

        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            from .. import global_config as config
            dist2_min_ek_out = torch.empty((E,H), device=dev, dtype=a_ik.dtype)
            kernel = KernelManager.real_enc_frame_proj_fwd_length
            kernel(((E * H + bsz - 1) // bsz,), (bsz,), (
                    _to_cupy(a_ik),
                    _to_cupy(rpos_ij_e),
                    _to_cupy(dist2_min_e),
                    _to_cupy(tvecs_n),
                    _to_cupy(batch_i),
                    _to_cupy(edge_ij_e),
                    to_uint32(N), to_uint8(H), to_uint32(E),
                    to_float32(dist_max), to_float32(wscale),
                    _to_cupy(W_k),
                    to_uint32(W_num),
                    _to_cupy(rvlen_n), to_float32(cutoff_radius),
                    _to_cupy(z_ek),
                    _to_cupy(v_ekd),
                    _to_cupy(dist2_min_ek_out)
            ))

            v_ekd *= length_rbf_mul
            if W_k1 is not None:
                frame_vecs = torch.stack([frame_vec_1,frame_vec_2,frame_vec_3], dim=0)
                W_ks = torch.stack([W_k1,W_k2,W_k3], dim=0)
                v_ekd_angle = torch.empty((3,)+v_ekd.shape, device=dev, dtype=v_ekd.dtype)
                kernel = KernelManager.real_enc_frame_proj_fwd_angle
                kernel(((3*E*H + bsz - 1) // bsz,), (bsz,), (
                        _to_cupy(a_ik),
                        _to_cupy(rpos_ij_e),
                        _to_cupy(dist2_min_ek_out),
                        _to_cupy(tvecs_n),
                        _to_cupy(batch_i),
                        _to_cupy(edge_ij_e),
                        to_uint32(N), to_uint8(H), to_uint32(E),
                        to_float32(angle_sigma),
                        _to_cupy(W_ks), _to_cupy(frame_vecs),
                        to_uint32(W_num),
                        _to_cupy(rvlen_n), to_float32(cutoff_radius),
                        _to_cupy(None),
                        _to_cupy(v_ekd_angle)
                ))
                v_ekd += v_ekd_angle.sum(dim=0) * angle_rbf_mul

        ctx.save_for_backward(a_ik, rpos_ij_e, dist2_min_e, tvecs_n, batch_i, edge_ij_e,
                              W_k, W_k1, W_k2, W_k3, frame_vec_1, frame_vec_2, frame_vec_3, z_ek, v_ekd)
        ctx.K = K
        ctx.rvlen_n = rvlen_n
        ctx.dist_max = float(dist_max)
        ctx.wscale = wscale
        ctx.cutoff_radius = cutoff_radius
        ctx.angle_sigma = angle_sigma
        ctx.length_rbf_mul = length_rbf_mul
        ctx.angle_rbf_mul = angle_rbf_mul
        if K <= 0:
            return z_ek,

        return z_ek, v_ekd

    @staticmethod
    def backward(ctx, gz_ek, gv_ekd=None):
        a_ik, rpos_ij_e, dist2_min_e, tvecs_n, batch_i, edge_ij_e, W_k, W_k1, W_k2, W_k3, frame_vec_1, frame_vec_2, frame_vec_3, z_ek, v_ekd = ctx.saved_tensors
        K = ctx.K
        rvlen_n = ctx.rvlen_n
        dist_max = ctx.dist_max
        wscale = ctx.wscale
        cutoff_radius = ctx.cutoff_radius
        N, H = a_ik.shape
        E = edge_ij_e.shape[1]

        length_rbf_mul = ctx.length_rbf_mul
        angle_rbf_mul = ctx.angle_rbf_mul

        angle_sigma = ctx.angle_sigma

        e_start_i = torch.zeros(N + 1, dtype=batch_i.dtype, device=batch_i.device)
        e_start_i.scatter_add_(0, edge_ij_e[0] + 1, torch.ones_like(edge_ij_e[0]))
        e_start_i = e_start_i.cumsum(0)

        ga_ik = torch.empty_like(a_ik, dtype = torch.float32)

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

        bsz = KernelManager.PE_THREAD_NUM
        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            from .. import global_config as config
            kernel = KernelManager.real_enc_frame_proj_bwd_length
            kernel(((N * H + bsz - 1) // bsz,), (bsz,), (
                _to_cupy(a_ik.detach()),
                _to_cupy(rpos_ij_e),
                _to_cupy(tvecs_n),
                _to_cupy(batch_i),
                _to_cupy(edge_ij_e),
                _to_cupy(e_start_i),
                _to_cupy(z_ek.detach()),
                _to_cupy(gz_ek.detach().contiguous()),
                _to_cupy(gv_ekd),
                to_uint32(N), to_uint8(H), to_uint32(E),  # K,
                to_float32(dist_max), to_float32(wscale),
                _to_cupy(W_k),
                to_uint32(W_num), _to_cupy(rvlen_n), to_float32(cutoff_radius),
                _to_cupy(ga_ik),
                _to_cupy(gW_k)
            ))

            ga_ik *= length_rbf_mul
            gW_k1 = gW_k2 = gW_k3 = None
            if W_k1 is not None:
                frame_vecs = torch.stack([frame_vec_1,frame_vec_2,frame_vec_3], dim=0)
                W_ks = torch.stack([W_k1,W_k2,W_k3], dim=0)
                gW_ks = torch.empty((3,max(W_num, N)) + W_ks.shape[-3:], device=dev, dtype=a_ik.dtype)
                ga_ik3 = torch.empty((3,)+ga_ik.shape, device=dev, dtype=a_ik.dtype)
                kernel = KernelManager.real_enc_frame_proj_bwd_angle
                kernel(((3*N*H + bsz - 1) // bsz,), (bsz,), (
                    _to_cupy(a_ik.detach()),
                    _to_cupy(rpos_ij_e),
                    _to_cupy(tvecs_n),
                    _to_cupy(batch_i),
                    _to_cupy(edge_ij_e),
                    _to_cupy(e_start_i),
                    _to_cupy(z_ek.detach()),
                    _to_cupy(gz_ek.detach().contiguous()),
                    _to_cupy(gv_ekd),
                    to_uint32(N), to_uint8(H), to_uint32(E),  # K,
                    to_float32(angle_sigma),
                    _to_cupy(W_ks),
                    _to_cupy(frame_vecs.detach().contiguous()),
                    to_uint32(W_num), _to_cupy(rvlen_n), to_float32(cutoff_radius),
                    _to_cupy(ga_ik3),
                    _to_cupy(gW_ks)
                ))
                ga_ik += ga_ik3.sum(dim=0) * angle_rbf_mul
                gW_k1 = gW_ks[0] * angle_rbf_mul
                gW_k2 = gW_ks[1] * angle_rbf_mul
                gW_k3 = gW_ks[2] * angle_rbf_mul

            gFrameVec1 = None
            gFrameVec2 = None
            gFrameVec3 = None

        if rvlen_n is None:
            return ga_ik, None, None, None, None, None, None, None, None, gW_k, gW_k1, gW_k2, gW_k3, gFrameVec1, gFrameVec2, gFrameVec3, None, None, None, None, None, None, None

        return ga_ik, None, None, None, None, None, None, None, None, gW_k, gW_k1, gW_k2, gW_k3, gFrameVec1, gFrameVec2, gFrameVec3, None, None, None, None, None, None, None, None, None, None





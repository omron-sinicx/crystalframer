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


class FusedDotProductAttentionSigmaMatCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, que_ihk, key_ihk, val_ihk, aij_eh, bij_ehk, batch_i, edge_ij_e):
        N, H, K = que_ihk.shape
        E = edge_ij_e.shape[1]
        dev = que_ihk.device

        e_start_i = torch.zeros(N + 1, dtype=batch_i.dtype, device=batch_i.device)
        e_start_i.scatter_add_(0, edge_ij_e[0] + 1, torch.ones_like(edge_ij_e[0]))
        e_start_i = e_start_i.cumsum(0)

        que_ihk = que_ihk.contiguous().detach()
        key_ihk = key_ihk.contiguous().detach()
        val_ihk = val_ihk.contiguous().detach()
        aij_eh = aij_eh.contiguous().detach() if aij_eh is not None else None
        bij_ehk = bij_ehk.contiguous().detach() if bij_ehk is not None else None
        batch_i = batch_i.contiguous()
        edge_ij_e = edge_ij_e.contiguous()

        output = torch.empty_like(val_ihk)
        prob_eh = torch.empty((E, H), dtype=que_ihk.dtype, device=dev)
        bsz = H
        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            kernel = KernelManager.fused_dpa_sigmamat_fwd
            kernel(((N * H + bsz - 1) // bsz,), (bsz,),
                   (
                       _to_cupy(que_ihk),
                       _to_cupy(key_ihk),
                       _to_cupy(val_ihk),
                       _to_cupy(aij_eh),
                       _to_cupy(bij_ehk),
                       _to_cupy(edge_ij_e),
                       _to_cupy(e_start_i),
                       N, H, E,
                       _to_cupy(prob_eh),
                       _to_cupy(output),
                   )
                   )
        ctx.save_for_backward(que_ihk, key_ihk, val_ihk, aij_eh, bij_ehk,
                              batch_i, edge_ij_e, e_start_i,
                              prob_eh, output)
        return output

    @staticmethod
    def backward(ctx, go_ihk):

        return None, None, None, None, None, None, None

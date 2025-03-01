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

class FusedDotProductAttentionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, que_ihk, key_ihk, val_ihk, aij_eh, bij_ehk, batch_i, edge_ij_e):
        N, H, K = que_ihk.shape
        E = edge_ij_e.shape[1]
        dev = que_ihk.device

        e_start_i = torch.zeros(N+1, dtype=batch_i.dtype, device=batch_i.device)
        e_start_i.scatter_add_(0, edge_ij_e[0]+1, torch.ones_like(edge_ij_e[0]))
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
            from .. import global_config as config
            kernel = KernelManager.fused_dpa_fwd_v3 if config.REPRODUCIBLITY_STATE>=3 \
                else KernelManager.fused_dpa_fwd
            kernel(((N*H+bsz-1)//bsz, ), (bsz, ),
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
        que_ihk, key_ihk, val_ihk, aij_eh, bij_ehk, \
        batch_i, edge_ij_e, e_start_i, \
        prob_eh, output = ctx.saved_tensors

        N, H, K = que_ihk.shape
        E = edge_ij_e.shape[1]
        dev = que_ihk.device

        B = batch_i.max().item()+1
        sizes = torch.zeros(B, dtype=torch.long, device=dev)
        sizes.scatter_add_(0, batch_i, torch.ones_like(batch_i))
        sizes2 = sizes*sizes

        gque = torch.empty_like(que_ihk)
        gkey = torch.empty_like(key_ihk)
        gval = torch.empty_like(val_ihk)
        gaij = torch.empty_like(aij_eh)
        gbij = torch.empty_like(bij_ehk) if bij_ehk is not None else None
        go_ihk = go_ihk.contiguous().detach()
        
        tprob_eh = torch.empty_like(prob_eh)
        tbij_ehk = torch.empty_like(bij_ehk) if bij_ehk is not None else None
        start_inds = torch.constant_pad_nd(sizes2.cumsum(0), (1,0))

        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            _start_inds = _to_cupy(start_inds)
            _sizes = _to_cupy(sizes)

            upper_mask = edge_ij_e[0] <= edge_ij_e[1]
            hE = upper_mask.long().sum().item()
            upper_e_t = torch.arange(E, dtype=torch.long, device=dev)[upper_mask]
            upper_batch_t = batch_i[edge_ij_e[0, upper_mask]]
            mat_sec_t = start_inds[upper_batch_t]
            sizes_t = sizes[upper_batch_t]

            def irregular_transpose(src:Tensor, dst:Tensor, C:int):
                bsz = min(32, C)
                KernelManager.irregular_transpose(
                    ((hE*C+bsz-1)//bsz, ), (bsz, ),
                    (_to_cupy(src), _to_cupy(upper_e_t), _to_cupy(mat_sec_t), _to_cupy(sizes_t), hE, C, _to_cupy(dst))
                )

            # def irregular_transpose(src:Tensor, dst:Tensor, C:int):
            #     bsz = min(32, C)
            #     kernels['irregular_transpose_old'](
            #         ((B*C+bsz-1)//bsz, ), (bsz, ),
            #         (_to_cupy(src), _to_cupy(start_inds), _to_cupy(sizes), B, C, _to_cupy(dst))
            #     )

            irregular_transpose(prob_eh, tprob_eh, H)
            if bij_ehk is not None:
                irregular_transpose(bij_ehk, tbij_ehk, H*K)

            assert (sizes <= KernelManager.MAX_SYSTEM_SIZE).all(), "Increase MAX_SYSTEM_SIZE in KernelManager"
            bsz = H
            from .. import global_config as config
            kernel = KernelManager.fused_dpa_bwd_v3 if config.REPRODUCIBLITY_STATE>=3 \
                else KernelManager.fused_dpa_bwd
            kernel(((N*H+bsz-1)//bsz, ), (bsz, ),
                (
                    _to_cupy(que_ihk),
                    _to_cupy(key_ihk),
                    _to_cupy(val_ihk),
                    _to_cupy(aij_eh),
                    _to_cupy(tbij_ehk),
                    _to_cupy(batch_i),
                    _to_cupy(edge_ij_e),
                    _to_cupy(e_start_i),
                    N, H, E,
                    _to_cupy(tprob_eh),
                    _to_cupy(output),
                    _to_cupy(go_ihk),
                    _to_cupy(gque),
                    _to_cupy(gkey),
                    _to_cupy(gval),
                    _to_cupy(gaij),
                    _to_cupy(gbij),
                )
            )

            # tranpose gaij and gbij
            irregular_transpose(gaij, gaij, H)
            if gbij is not None:
                irregular_transpose(gbij, gbij, H*K)
            
            # use gaij as grad softmax to compute grad q.
            bsz = H
            kernel = KernelManager.fused_dpa_bwd_q_v3 if config.REPRODUCIBLITY_STATE>=3 \
                else KernelManager.fused_dpa_bwd_q
            kernel(((N*H+bsz-1)//bsz, ), (bsz, ),
                (
                    _to_cupy(key_ihk),
                    _to_cupy(gaij),
                    _to_cupy(edge_ij_e),
                    _to_cupy(e_start_i),
                    N, H, E,
                    _to_cupy(gque),
                )
            )


        return gque, gkey, gval, gaij, gbij, None, None

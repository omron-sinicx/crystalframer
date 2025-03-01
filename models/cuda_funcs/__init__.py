
CUPY_AVAILABLE=True

try:
    import cupy as cp
    import pytorch_pfn_extras as ppe
    from torch.utils.dlpack import to_dlpack, from_dlpack
    ppe.cuda.use_torch_mempool_in_cupy()
except:
    CUPY_AVAILABLE = False

from .kernel_manager import Kernel, KernelManager, compile_kernels
from .real_space_enc import RealPeriodicEncodingFuncCUDA
from .real_space_enc_proj import RealPeriodicEncodingWithProjFuncCUDA
from .real_space_framed_proj import RealEncodingWithFramedProjFuncCUDA
from .reci_space_enc import ReciPeriodicEncodingFuncCUDA
from .calc_maximum_frame import ComputeFrame1MaximumMethodCUDA
from .calc_maximum_static_frame import ComputeFrame1MaximumStaticMethodCUDA
from .calc_maximum_static_frame import ComputeFrame2MaximumStaticMethodCUDA
from .calc_maximum_frame import ComputeFrame2MaximumMethodCUDA
from .calc_maximum_frame import ComputeFrame3MaximumMethodCUDA
from .calc_lattice_frame import ComputeLatticeFrameCUDA
from .calc_pca_frame import ComputePCAFrame
from .fused_dpa import FusedDotProductAttentionCUDA
from .fused_dpa_sigmamat import FusedDotProductAttentionSigmaMatCUDA
from .irregular_mean import IrregularMeanCUDA

__all__ = [
    'KernelManager', 
    'Kernel',
    'compile_kernels',
    'FusedDotProductAttentionCUDA',
    'FusedDotProductAttentionSigmaMatCUDA',
    'ComputeFrame1MaximumMethodCUDA',
    'ComputeFrame1MaximumStaticMethodCUDA',
    'ComputeFrame2MaximumMethodCUDA',
    'ComputeFrame2MaximumStaticMethodCUDA',
    'ComputeFrame3MaximumMethodCUDA',
    'RealPeriodicEncodingFuncCUDA',
    'RealPeriodicEncodingWithProjFuncCUDA',
    'RealEncodingWithFramedProjFuncCUDA',
    'ReciPeriodicEncodingFuncCUDA',
    'ComputeLatticeFrameCUDA',
    'ComputePCAFrame',
    'IrregularMeanCUDA',
    'CUPY_AVAILABLE',
]

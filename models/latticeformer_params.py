from typing import TypeVar, Type, List
import copy
Entity = TypeVar('Entity', bound='LatticeformerParams')

class LatticeformerParams:
    def __init__(self, 
                 domain:str="real",
                 lattice_range:int=4,
                 minimum_range:bool=True,
                 adaptive_cutoff_sigma:float=-3.5,
                 gauss_lb_real:float=0.5,
                 gauss_lb_reci:float=0.5,
                 scale_real:List[float]=[1.4],
                 scale_reci:List[float]=[2.2],
                 normalize_gauss:bool=True,
                 value_pe_dist_real:int=64,
                 value_pe_dist_coef:float=1.0,
                 value_pe_dist_max:float=-10.0,
                 value_pe_dist_wscale:float=1.0,
                 value_pe_wave_real:int=0,
                 value_pe_dist_reci:int=0,
                 value_pe_wave_reci:int=0,
                 value_pe_angle_real:int = 16,
                 value_pe_angle_coef:float = 1.0,
                 value_pe_angle_wscale:float = 4.0,
                 positive_func_beta:float=0.1,
                 layer_index:int=-1,
                 gauss_state:str="q",
                 frame_method: str="max",
                 frame_mode:str="both",
                 cos_abs: int = 1,
                 symm_break_noise:float = 1e-5,
                 ) -> None:

        self.layer_index = layer_index
        self.domain = domain
        self.cos_abs = cos_abs
        self.lattice_range = lattice_range
        self.minimum_range = minimum_range
        self.adaptive_cutoff_sigma = adaptive_cutoff_sigma
        self.gauss_lb_real = gauss_lb_real
        self.gauss_lb_reci = gauss_lb_reci
        self.scale_real = scale_real
        self.scale_reci = scale_reci
        self.normalize_gauss = normalize_gauss
        self.value_pe_dist_real = value_pe_dist_real
        self.value_pe_dist_coef = value_pe_dist_coef
        self.value_pe_dist_max = value_pe_dist_max
        self.value_pe_dist_wscale = value_pe_dist_wscale
        self.value_pe_wave_real = value_pe_wave_real
        self.value_pe_dist_reci = value_pe_dist_reci
        self.value_pe_wave_reci = value_pe_wave_reci
        self.value_pe_angle_real = value_pe_angle_real
        self.value_pe_angle_coef = value_pe_angle_coef
        self.value_pe_angle_wscale = value_pe_angle_wscale
        self.positive_func_beta = positive_func_beta
        self.gauss_state = gauss_state
        self.frame_mode = frame_mode
        self.frame_method = frame_method
        self.symm_break_noise = symm_break_noise

    def parseFromArgs(self, args):
        for key in self.__dict__:
            self.__dict__[key] = getattr(args, key, self.__dict__[key])
        print("Parsed LatticeformerParams:")
        print(self.__dict__)

    def getLayerParameters(self, layer_index) -> Entity:
        if self.domain in ("real", "reci", "multihead"):
            domain = self.domain
        else:
            domains = self.domain.split('-')
            domain = domains[layer_index % len(domains)]

        scale_real = self.scale_real
        scale_reci = self.scale_reci
        if isinstance(scale_real, (list,tuple)):
            scale_real = scale_real[layer_index % len(scale_real)]
        if isinstance(scale_reci, (list,tuple)):
            scale_reci = scale_reci[layer_index % len(scale_reci)]

        params = copy.deepcopy(self)
        params.domain = domain
        params.scale_real = scale_real
        params.scale_reci = scale_reci
        params.layer_index = layer_index
        return params
    
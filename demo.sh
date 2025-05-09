#!/usr/bin/bash

save_path="result/crystalframer/JARVIS"
weights_path="model weights path (.ckpt)"

# | target_set                      | targets |
# jarvis__megnet                    | e_form | bandgap |
# jarvis__megnet-bulk               | bulk_modulus
# jarvis__megnet-shear              | shear_modulus
# jarvis__dft_3d_2021               | formation_energy | total_energy | opt_bandgap |
# jarvis__dft_3d_2021-ehull         | ehull |
# jarvis__dft_3d_2021-mbj_bandgap   | mbj_bandgap |
# jarvis__oqmd_3d                   | stability | delta_e |
# jarvis__oqmd_3d-bandgap           | bandgap |

targets=formation_energy
target_set=jarvis__dft_3d_2021
frame_method=max  #max, weighted_pca, max_static, lattice, pca
exp_name=max
gpu=0


CUDA_VISIBLE_DEVICES=${gpu} python demo.py -p latticeformer/default_jarvis.json \
    --frame_method ${frame_method} \
    --value_pe_dist_coef 1.0 \
    --value_pe_angle_wscale 4.0 \
    --value_pe_angle_real 64 \
    --value_pe_angle_coef 4.0 \
    --save_path ${save_path} \
    --domain real \
    --num_layers 4 \
    --batch_size 256 \
    --experiment_name ${exp_name}/${targets} \
    --target_set ${target_set} \
    --targets ${targets} \
    --pretrained_model ${weights_path} \


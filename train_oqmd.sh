#!/usr/bin/bash

save_path="result/crystalframer/OQMD"

# | target_set                      | targets |
# jarvis__megnet                    | e_form | bandgap |
# jarvis__megnet-bulk               | bulk_modulus
# jarvis__megnet-shear              | shear_modulus
# jarvis__dft_3d_2021               | formation_energy | total_energy | opt_bandgap |
# jarvis__dft_3d_2021-ehull         | ehull |
# jarvis__dft_3d_2021-mbj_bandgap   | mbj_bandgap |
# jarvis__oqmd_3d                   | stability | delta_e |
# jarvis__oqmd_3d-bandgap           | bandgap |

targets=stability
target_set=jarvis__oqmd_3d
frame_method=max  #max, weighted_pca, max_static, lattice, pca
exp_name=max
gpu=0


CUDA_VISIBLE_DEVICES=${gpu} python train.py -p crystalframer/default_oqmd.json \
    --frame_method max \
    --value_pe_dist_coef 1.0 \
    --value_pe_angle_wscale 4.0 \
    --value_pe_angle_real 64 \
    --value_pe_angle_coef 1.0 \
    --save_path ${save_path} \
    --domain real \
    --num_layers 4 \
    --batch_size 1024 \
    --train_filter_max 320 \
    --experiment_name ${exp_name}/${targets} \
    --target_set ${target_set} \
    --targets ${targets} \



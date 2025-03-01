#include <math_constants.h>
extern "C" __global__

void real_enc_frame_proj_fwd(
    const float* a_ik,
    const float* rpos_ij_e,
    const float* dist2_min_e,
    const float* tvecs_n,
    const long long int* batch_i,
    const long long int* edge_ij_e,
    const long long int N,
    const long long int H,
    const long long int E,
    const long long int K_,
    const double dist_max,
    const double wscale,
    const double angle_sigma,
    const float* W_k1,
    const float* W_k2,
    const float* W_k3,
    const float* W_k,
    const float* frame_vec,
    const long long int W_num,
    const float* rveclens_n,
    const double cutoff_radius,
    const long long int mode,
    float* z_ek,
    float* v_ekd){
    const long long int tid = (long long int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= E*H) return;

    const long long int k = tid%H;
    const long long int e = tid/H;
    const long long int i = edge_ij_e[e];
    const long long int j = edge_ij_e[E+e];
    const long long int n = batch_i[i];
    rpos_ij_e += e*3;
    const float r_ijx = rpos_ij_e[0];
    const float r_ijy = rpos_ij_e[1];
    const float r_ijz = rpos_ij_e[2];
    tvecs_n += n*9;
    const float t1_x = tvecs_n[0];
    const float t1_y = tvecs_n[1];
    const float t1_z = tvecs_n[2];
    const float t2_x = tvecs_n[3];
    const float t2_y = tvecs_n[4];
    const float t2_z = tvecs_n[5];
    const float t3_x = tvecs_n[6];
    const float t3_y = tvecs_n[7];
    const float t3_z = tvecs_n[8];
    const float a = a_ik[i*H + k];
    const int R = LATTICE_RANGE;
    const float Rf = (float)LATTICE_RANGE;

    #if VPE_DIM > 0

    __shared__ float shared_v[THREAD_NUM][VPE_DIM+1];
    float *sv = shared_v[threadIdx.x];
    __shared__ float shared_v_angle1[THREAD_NUM][ANGLE_ENC_DIM+1];
    float *sv_angle1 = shared_v_angle1[threadIdx.x];
    __shared__ float shared_v_angle2[THREAD_NUM][ANGLE_ENC_DIM+1];
    float *sv_angle2 = shared_v_angle2[threadIdx.x];
    __shared__ float shared_v_angle3[THREAD_NUM][ANGLE_ENC_DIM+1];
    float *sv_angle3 = shared_v_angle3[threadIdx.x];
    for (int dim = 0; dim < VPE_DIM; dim++)
    {
        sv[dim] = 0;
    }
    for (int dim = 0; dim < ANGLE_ENC_DIM; dim++)
    {
        sv_angle1[dim] = 0;
        sv_angle2[dim] = 0;
        sv_angle3[dim] = 0;
    }

    const float reci_ws_sqrt2 = 1.0f/((float)wscale*sqrt(2.0f));
    const float mu0 = (float)dist_max/VPE_DIM;
    const float sig = mu0 * wscale;
    #endif

    rveclens_n += n*3;
    const float rvl1 = rveclens_n[0];
    const float rvl2 = rveclens_n[1];
    const float rvl3 = rveclens_n[2];

    float cutoff = (float)cutoff_radius;
    int R1 = LATTICE_RANGE, R2 = LATTICE_RANGE, R3 = LATTICE_RANGE;
    if (cutoff != 0.0f)
    {
        if (cutoff < 0) {
            // Better sync the threads in each block?
            // -> disabled due to thread stucking
            // float a_max = a;
            // for (int t = 0; t < THREAD_NUM; t++)
            //     a_max = max(a_max, a_ik[i*H + t]);
            //cutoff = sqrt(-0.5f/a_max)*(-cutoff);
            cutoff = sqrt(-0.5f/a)*(-cutoff);
        }
        R1 = ceil((cutoff + 0.01f)*rvl1/(2.0*CUDART_PI_F));
        R2 = ceil((cutoff + 0.01f)*rvl2/(2.0*CUDART_PI_F));
        R3 = ceil((cutoff + 0.01f)*rvl3/(2.0*CUDART_PI_F));

        #if MINIMUM_RANGE > 0
        R1 = max(R1, MINIMUM_RANGE);
        R2 = max(R2, MINIMUM_RANGE);
        R3 = max(R3, MINIMUM_RANGE);
        #endif
    }

    float d2min = 1e10;
    if (1 || dist2_min_e == NULL)
    {
        for (float n1 = -R1; n1 <= R1; n1++)
        for (float n2 = -R2; n2 <= R2; n2++)
        for (float n3 = -R3; n3 <= R3; n3++)
        {
            float dx = r_ijx + t1_x*n1 + t2_x*n2 + t3_x*n3;
            float dy = r_ijy + t1_y*n1 + t2_y*n2 + t3_y*n3;
            float dz = r_ijz + t1_z*n1 + t2_z*n2 + t3_z*n3;
            float d2 = dx*dx + dy*dy + dz*dz;
            // float dx = fmaf(t1_x, n1, fmaf(t2_x, n2, fmaf(t3_x, n3, r_ijx)));
            // float dy = fmaf(t1_y, n1, fmaf(t2_y, n2, fmaf(t3_y, n3, r_ijy)));
            // float dz = fmaf(t1_z, n1, fmaf(t2_z, n2, fmaf(t3_z, n3, r_ijz)));
            // float d2 = fmaf(dx,dx, fmaf(dy,dy, dz*dz));
            d2min = fminf(d2min, d2);
        }
    } else {
        d2min = dist2_min_e[e];
    }

    #if COS_ABS == 1
    float step_angle = 1.0f/((ANGLE_ENC_DIM - 1) * angle_sigma * sqrtf(2));
    #endif
    #if COS_ABS == 0
    float step_angle = 2.0f/((ANGLE_ENC_DIM - 1) * angle_sigma * sqrtf(2));
    #endif

    float sum = 0;

    double angle_sigma_inv = 1 / (angle_sigma * sqrtf(2));

    for (float n1 = -R1; n1 <= R1; n1++)
    for (float n2 = -R2; n2 <= R2; n2++)
    for (float n3 = -R3; n3 <= R3; n3++)
    {
        float dx = r_ijx + t1_x*n1 + t2_x*n2 + t3_x*n3;
        float dy = r_ijy + t1_y*n1 + t2_y*n2 + t3_y*n3;
        float dz = r_ijz + t1_z*n1 + t2_z*n2 + t3_z*n3;
        float d2 = dx*dx + dy*dy + dz*dz;
        // float dx = fmaf(t1_x, n1, fmaf(t2_x, n2, fmaf(t3_x, n3, r_ijx)));
        // float dy = fmaf(t1_y, n1, fmaf(t2_y, n2, fmaf(t3_y, n3, r_ijy)));
        // float dz = fmaf(t1_z, n1, fmaf(t2_z, n2, fmaf(t3_z, n3, r_ijz)));
        // float d2 = fmaf(dx,dx, fmaf(dy,dy, dz*dz));

        int index = edge_ij_e[blockIdx.x];

        float cos_angle_1 = 1e10;
        float cos_angle_2 = 1e10;
        float cos_angle_3 = 1e10;
        if (d2 != 0.0f){
        cos_angle_1 = (dx*frame_vec[index*9*H+9*threadIdx.x+0] + dy*frame_vec[index*9*H+9*threadIdx.x+3] + dz*frame_vec[index*9*H+9*threadIdx.x+6]) / sqrtf(d2);
        cos_angle_2 = (dx*frame_vec[index*9*H+9*threadIdx.x+1] + dy*frame_vec[index*9*H+9*threadIdx.x+4] + dz*frame_vec[index*9*H+9*threadIdx.x+7]) / sqrtf(d2);
        cos_angle_3 = (dx*frame_vec[index*9*H+9*threadIdx.x+2] + dy*frame_vec[index*9*H+9*threadIdx.x+5] + dz*frame_vec[index*9*H+9*threadIdx.x+8]) / sqrtf(d2);
        }

        float w = expf(a*(d2 - d2min));
        sum += w;

        #if VPE_DIM > 0
        // b_dim = exp( -((dim*(m/K)-dist)/(sqrt(2)*wscale*dist_max/K))**2 )
        float b = -sqrtf(d2)/mu0*reci_ws_sqrt2;

        #if COS_ABS == 1
        float angle_1_tmp = (-fabsf(cos_angle_1) + 1.0f) * angle_sigma_inv;
        float angle_2_tmp = (-fabsf(cos_angle_2) + 1.0f) * angle_sigma_inv;
        float angle_3_tmp = (-fabsf(cos_angle_3) + 1.0f) * angle_sigma_inv;
        #endif

        #if COS_ABS == 0
        float angle_1_tmp = (-cos_angle_1 + 1.0f) * angle_sigma_inv;
        float angle_2_tmp = (-cos_angle_2 + 1.0f) * angle_sigma_inv;
        float angle_3_tmp = (-cos_angle_3 + 1.0f) * angle_sigma_inv;
        #endif

        #pragma unroll
        for (int dim = 0; dim < VPE_DIM; dim++)
        {
            b += reci_ws_sqrt2;
            //if (tid == 0 && n1 == 1 && n2 == 0 && n3 == 0) printf("%f\n",exp(-b*b));
            sv[dim] += exp(-b*b)*w;
        }

        #pragma unroll
        for (int dim = 0; dim < ANGLE_ENC_DIM; dim++)
        {
            sv_angle1[ANGLE_ENC_DIM-1-dim] += exp(-angle_1_tmp*angle_1_tmp)*w;
            sv_angle2[ANGLE_ENC_DIM-1-dim] += exp(-angle_2_tmp*angle_2_tmp)*w;
            sv_angle3[ANGLE_ENC_DIM-1-dim] += exp(-angle_3_tmp*angle_3_tmp)*w;
            //if (tid == 0 && n1 == 1 && n2 == 0 && n3 == 0) printf("%f\n", exp(-angle_1_tmp*angle_1_tmp));
            angle_1_tmp -= step_angle;
            angle_2_tmp -= step_angle;
            angle_3_tmp -= step_angle;

        }
        #endif
        /*
        add-dest   exp
        shared-m    X   389.8009338378906 ± 21.061248779296875
        register    X   345.8869018554688 ± 20.871074676513672
        shared-m    -   398.120849609375  ± 21.012807846069336
        register    -   268.1198425292969 ± 20.880420684814453
        */
    }

    #if VPE_DIM > 0
    if (W_k == NULL){
        float *v = &v_ekd[tid*VPE_DIM];
        #pragma unroll
        for (int dim = 0; dim < VPE_DIM; dim++)
            v[dim] = sv[dim]/sum;
    } else {
        // Do the matrix-vector multiplication: v' = Wv.
        float *v = &v_ekd[tid*V_HEAD_DIM];
        long long int w_ind = 0;
        if (W_num == 1){
            w_ind = 0;
        } else if (W_num == E) {
            w_ind = e;
        } else if (W_num == N) {
            w_ind = i;
        }
        const float *W = &W_k[(w_ind*H+k)*V_HEAD_DIM*VPE_DIM];
        const float *W1 = &W_k1[(w_ind*H*k)*V_HEAD_DIM*ANGLE_ENC_DIM];
        const float *W2 = &W_k1[(w_ind*H*k)*V_HEAD_DIM*ANGLE_ENC_DIM];
        const float *W3 = &W_k1[(w_ind*H*k)*V_HEAD_DIM*ANGLE_ENC_DIM];


        for (int wdim = 0; wdim < V_HEAD_DIM; wdim++){
            float sum_v = 0;
            if (mode == 1){
            #pragma unroll
            for (int dim = 0; dim < VPE_DIM; dim++){
                // For numerical accuracy, it is important to do "sv[dim]/sum"
                // instead of "sum_v/sum" after the loop.

                sum_v += W[wdim*VPE_DIM+dim]*(sv[dim]);
            }
            }
            #pragma unroll
            for (int dim = 0; dim < ANGLE_ENC_DIM; dim++){
                sum_v += W1[wdim*ANGLE_ENC_DIM+dim]*(sv_angle1[dim]);
            }
            #pragma unroll
            for (int dim = 0; dim < ANGLE_ENC_DIM; dim++){
                sum_v += W2[wdim*ANGLE_ENC_DIM+dim]*(sv_angle2[dim]);
            }
            #pragma unroll
            for (int dim = 0; dim < ANGLE_ENC_DIM; dim++){
                sum_v += W3[wdim*ANGLE_ENC_DIM+dim]*(sv_angle3[dim]);
            }
            v[wdim] = sum_v/sum;
        }
    }
    #endif

    z_ek[tid] = logf(sum) + d2min*a;
}
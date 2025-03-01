#include <math_constants.h>
extern "C" __global__

void real_enc_frame_proj_bwd(
    const float* a_ik,
    const float* rpos_ij_e,
    //const float* dist2_min_e,
    const float* tvecs_n,
    const long long int* batch_i,
    const long long int* edge_ij_e,
    const long long int* e_start_i,
    const float* z_ek,
    const float* gz_ek,
    const float* gv_ekd,
    const long long int N,
    const long long int H,
    const long long int E,
    //const long long int K_,
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
    float* ga_ik,
    float* gW_k,
    float* gW_k1,
    float* gW_k2,
    float* gW_k3){
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N*H) return;

    const unsigned int k = tid%H;
    const unsigned int i = tid/H;
    const unsigned int n = batch_i[i];
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
    const unsigned int e_end = e_start_i[i+1];
    #if VPE_DIM > 0
    __shared__ float shared_gv[THREAD_NUM][VPE_DIM+1];
    __shared__ float shared_v[THREAD_NUM][VPE_DIM+1];
    float *sv = shared_v[threadIdx.x];
    float *gW = NULL;
    if (gW_k != NULL && (W_num == N || W_num == 1)){
        gW = &gW_k[(i*H+k)*V_HEAD_DIM*VPE_DIM];
        for (int dim = 0; dim < V_HEAD_DIM*VPE_DIM; dim++)
            gW[dim] = 0;
    }
    #endif
    __shared__ float shared_v_angle1[THREAD_NUM][ANGLE_ENC_DIM+1];
    __shared__ float shared_v_angle2[THREAD_NUM][ANGLE_ENC_DIM+1];
    __shared__ float shared_v_angle3[THREAD_NUM][ANGLE_ENC_DIM+1];
    float *sv_angle1 = shared_v_angle1[threadIdx.x];
    float *sv_angle2 = shared_v_angle2[threadIdx.x];
    float *sv_angle3 = shared_v_angle3[threadIdx.x];
    float *gW_angle1 = NULL;
    float *gW_angle2 = NULL;
    float *gW_angle3 = NULL;
    if (gW_k1 != NULL && (W_num == N || W_num == 1)){
        gW_angle1 = &gW_k1[(i*H+k)*V_HEAD_DIM*ANGLE_ENC_DIM];
        gW_angle2 = &gW_k2[(i*H+k)*V_HEAD_DIM*ANGLE_ENC_DIM];
        gW_angle3 = &gW_k3[(i*H+k)*V_HEAD_DIM*ANGLE_ENC_DIM];
        for (int dim = 0; dim < V_HEAD_DIM*ANGLE_ENC_DIM; dim++){
            gW_angle1[dim] = 0;
            gW_angle2[dim] = 0;
            gW_angle3[dim] = 0;
        }
    }

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
        float cutoff2 = cutoff*cutoff;

        #if MINIMUM_RANGE > 0
        R1 = max(R1, MINIMUM_RANGE);
        R2 = max(R2, MINIMUM_RANGE);
        R3 = max(R3, MINIMUM_RANGE);
        #endif
    }

    float sum = 0;
    float sum_v = 0;
    for (unsigned int e = e_start_i[i]; e < e_end; e++)
    {
        const unsigned int j = edge_ij_e[E+e];
        const float r_ijx = rpos_ij_e[e*3+0];
        const float r_ijy = rpos_ij_e[e*3+1];
        const float r_ijz = rpos_ij_e[e*3+2];
        const unsigned int ek = e*H+k;
        const float z = z_ek[ek];
        const float gz = gz_ek[ek];

        #if VPE_DIM > 0
        float *sgv = shared_gv[threadIdx.x];

        if (gW_k == NULL){
            const float *gv = &gv_ekd[ek*VPE_DIM];
            #pragma unroll
            for (int dim = 0; dim < VPE_DIM; dim++) {
                sgv[dim] = gv[dim];
            }
        } else {
            // Compute backward of v' = Wv, as gW = (gv')^T * v
            const float *gv = &gv_ekd[ek*V_HEAD_DIM];
            unsigned int w_ind = 0;
            if (W_num == 1){
                w_ind = 0;
            } else if (W_num == E) {
                w_ind = e;
            } else if (W_num == N) {
                w_ind = i;
            }
            const float *W = &W_k[(w_ind*H+k)*V_HEAD_DIM*VPE_DIM];
            #pragma unroll
            for (int dim = 0; dim < VPE_DIM; dim++)
                sgv[dim] = 0;
            #pragma unroll
            for (int wdim = 0; wdim < V_HEAD_DIM; wdim++){
                float gv_val = gv[wdim];
                #pragma unroll
                for (int dim = 0; dim < VPE_DIM; dim++){
                    sgv[dim] += W[wdim*VPE_DIM+dim]*gv_val;
                    //sgv[dim] += (*W++)*gv_val;
                }
            }

            // for gW
            if (W_num == E){
                gW = &gW_k[(e*H+k)*V_HEAD_DIM*VPE_DIM];
                for (int dim = 0; dim < V_HEAD_DIM*VPE_DIM; dim++)
                    gW[dim] = 0;
            }
        }
        #endif

        float px_avr = 0;
        float pbg_avr = 0;
        const float reci_ws_sqrt2 = 1.0f/((float)wscale*sqrt(2.0f));
        const float mu0 = (float)dist_max/VPE_DIM;

        #if COS_ABS == 0
        float step_angle = 2.0f/((ANGLE_ENC_DIM - 1) * angle_sigma * sqrtf(2));
        #endif
        #if COS_ABS == 1
        float step_angle = 1.0f/((ANGLE_ENC_DIM - 1) * angle_sigma * sqrtf(2));
        #endif

        #if VPE_DIM > 0
        if (gW_k != NULL){
            #pragma unroll
            for (int dim = 0; dim < VPE_DIM; dim++)
                sv[dim] = 0;
        }
        #endif
        #pragma unroll
        for (int dim = 0; dim < ANGLE_ENC_DIM; dim++)
        {
        sv_angle1[dim] = 0;
        sv_angle2[dim] = 0;
        sv_angle3[dim] = 0;
        }

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
            if (d2 != 0.0f)
            {
            cos_angle_1 = (dx*frame_vec[index*9*H+9*threadIdx.x+0] + dy*frame_vec[index*9*H+9*threadIdx.x+3] + dz*frame_vec[index*9*H+9*threadIdx.x+6]) / sqrtf(d2);
            cos_angle_2 = (dx*frame_vec[index*9*H+9*threadIdx.x+1] + dy*frame_vec[index*9*H+9*threadIdx.x+4] + dz*frame_vec[index*9*H+9*threadIdx.x+7]) / sqrtf(d2);
            cos_angle_3 = (dx*frame_vec[index*9*H+9*threadIdx.x+2] + dy*frame_vec[index*9*H+9*threadIdx.x+5] + dz*frame_vec[index*9*H+9*threadIdx.x+8]) / sqrtf(d2);
            }

            #if COS_ABS == 1
            int end_dim_angle1 = min((fabsf(cos_angle_1) + CUT_SIGMA*angle_sigma)/interval_angle + 1, (float)ANGLE_ENC_DIM);
            int end_dim_angle2 = min((fabsf(cos_angle_2) + CUT_SIGMA*angle_sigma)/interval_angle + 1, (float)ANGLE_ENC_DIM);
            int end_dim_angle3 = min((fabsf(cos_angle_3) + CUT_SIGMA*angle_sigma)/interval_angle + 1, (float)ANGLE_ENC_DIM);
            float angle_1_tmp = (-fabsf(cos_angle_1) + 1.0f) * angle_sigma_inv - (ANGLE_ENC_DIM - end_dim_angle1)*step_angle;
            float angle_2_tmp = (-fabsf(cos_angle_2) + 1.0f) * angle_sigma_inv - (ANGLE_ENC_DIM - end_dim_angle2)*step_angle;
            float angle_3_tmp = (-fabsf(cos_angle_3) + 1.0f) * angle_sigma_inv - (ANGLE_ENC_DIM - end_dim_angle3)*step_angle;
            #endif

            #if COS_ABS == 0
            float angle_1_tmp = (-cos_angle_1 + 1.0f) * angle_sigma_inv;
            float angle_2_tmp = (-cos_angle_2 + 1.0f) * angle_sigma_inv;
            float angle_3_tmp = (-cos_angle_3 + 1.0f) * angle_sigma_inv;
            int end_dim_angle1 = max((-cos_angle_1 + 1.0f + CUT_SIGMA*angle_sigma)/interval_angle + 1, -(float)BUFF);
            int end_dim_angle2 = max((-cos_angle_2 + 1.0f + CUT_SIGMA*angle_sigma)/interval_angle + 1, -(float)BUFF);
            int end_dim_angle3 = max((-cos_angle_3 + 1.0f + CUT_SIGMA*angle_sigma)/interval_angle + 1, -(float)BUFF);
            #endif

            float p = expf(a*d2 - z);
            float px = d2*p;
            px_avr += px;

            #if VPE_DIM > 0
            float bg = 0;
            float b = -sqrtf(d2)/mu0*reci_ws_sqrt2;
            #pragma unroll
            for (int dim = 0; dim < VPE_DIM; dim++)
            {
                b += reci_ws_sqrt2;
                float gauss = expf(-b*b);
                bg += gauss*sgv[dim];
                sv[dim] += gauss*p;
            }
            for (int dim = 0; dim < ANGLE_ENC_DIM; dim++)
            {
                sv_angle1[dim] += exp(-angle_1_tmp*angle_1_tmp)*p;
                sv_angle2[dim] += exp(-angle_2_tmp*angle_2_tmp)*p;
                sv_angle3[dim] += exp(-angle_3_tmp*angle_3_tmp)*p;
                angle_1_tmp -= step_angle;
                angle_2_tmp -= step_angle;
                angle_3_tmp -= step_angle;
            }
            sum_v += px*bg;
            pbg_avr += p*bg;
            #endif
        }
        /*
        b: (E, 1, R, K)
        x: (E, 1, R, 1)
        y: (N, H, 1, 1)
        z: (E, H, 1, K)
        g: (E, H, 1, K)
        p: (E, H, R, 1)

        (E,H,R,K)   (E,H,R,1)     (E,H,R,K)       (E,H,1,K): (E,H,R,1)*(E,1,R,K)*(E,H,1,K)
        dz/dye    =    p*x    * (    b*g     -    (p*b*g).sum(axis=R))

        (E,H,1,1)
        dz/dyi    = (dz/dye).sum(axis=R,K).sum_for_j()

                     (E,H,R,1)*(E,H,R,1)                (E,H,1,1)        *(E,H,1,1)
        dz/dye    =    (p*x)  *(b*g).sum(axis=K)    -   (p*x).sum(axis=R)*(p*b*g).sum(axis=R,K))
        */

        sum += px_avr*gz;
        sum_v -= px_avr*pbg_avr;

        #if VPE_DIM > 0
        if (gW_k != NULL){
            const float *gv = &gv_ekd[ek*V_HEAD_DIM];
            #pragma unroll
            for (int wdim = 0; wdim < V_HEAD_DIM; wdim++){
                float gv_val = gv[wdim];
                #pragma unroll
                for (int dim = 0; dim < VPE_DIM; dim++){
                    //*(_sgw++) += sv[dim]*gv_val;
                    gW[wdim*VPE_DIM+dim] += sv[dim]*gv_val;
                }
                #pragma unroll
                for (int dim = 0; dim < ANGLE_ENC_DIM; dim++){
                    //*(_sgw++) += sv[dim]*gv_val;
                    gW_angle1[wdim*ANGLE_ENC_DIM+dim] += sv_angle1[dim]*gv_val;
                    gW_angle2[wdim*ANGLE_ENC_DIM+dim] += sv_angle2[dim]*gv_val;
                    gW_angle3[wdim*ANGLE_ENC_DIM+dim] += sv_angle3[dim]*gv_val;
                }
            }
        }
        #endif
    }
    ga_ik[tid] = sum + sum_v;
}
#include <math_constants.h>
extern "C" __global__

// whether cache 'gv_ekd' in shared memory or load from global memory each time
#if PE_THREAD_NUM*(VPE_DIM+1)*2 + PE_THREAD_NUM*(V_HEAD_DIM+1) <= 1024
#define CACHE_GV 1
#else
#define CACHE_GV 0
#endif

void real_enc_frame_proj_bwd_length(
    const float* a_ik,
    const float* rpos_ij_e,
    const float* tvecs_n,
    const long long int* batch_i,
    const long long int* edge_ij_e,
    const long long int* e_start_i,
    const float* z_ek,
    const float* gz_ek,
    const float* gv_ekd,
    const unsigned int N,
    const unsigned char H,
    const unsigned int E,
    const float dist_max,
    const float wscale,
    const float* W_k,
    const unsigned int W_num,
    const float* rveclens_n,
    const float cutoff_radius,
    float* ga_ik,
    float* gW_k)
    {
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
    #if CACHE_GV
    __shared__ float shared_gv_ek[PE_THREAD_NUM][V_HEAD_DIM+1];
    #endif

    __shared__ float shared_gv[PE_THREAD_NUM][VPE_DIM+1];
    __shared__ float shared_v[PE_THREAD_NUM][VPE_DIM+1];

    #endif

    #if VPE_DIM > 0
    if (gW_k != NULL && (W_num == N || W_num == 1)){
        gW_k += (i*H+k)*V_HEAD_DIM*VPE_DIM;
        for (int dim = 0; dim < V_HEAD_DIM*VPE_DIM; dim++)
            gW_k[dim] = 0;
    }
    #endif

    float cutoff = (float)cutoff_radius;
    int R1 = LATTICE_RANGE, R2 = LATTICE_RANGE, R3 = LATTICE_RANGE;
    if (cutoff != 0.0f)
    {
        rveclens_n += n*3;
        const float rvl1 = rveclens_n[0];
        const float rvl2 = rveclens_n[1];
        const float rvl3 = rveclens_n[2];

        if (cutoff < 0) {
            // Better sync the threads in each block?
            // -> disabled due to thread stucking
            // float a_max = a;
            // for (int t = 0; t < PE_THREAD_NUM; t++)
            //     a_max = max(a_max, a_ik[i*H + t]);
            //cutoff = sqrt(-0.5f/a_max)*(-cutoff);
            cutoff = sqrt(-0.5f/a)*(-cutoff);
        }
        R1 = ceil((cutoff + 0.01f)*rvl1/(2.0*CUDART_PI_F));
        R2 = ceil((cutoff + 0.01f)*rvl2/(2.0*CUDART_PI_F));
        R3 = ceil((cutoff + 0.01f)*rvl3/(2.0*CUDART_PI_F));
        //float cutoff2 = cutoff*cutoff;

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
        unsigned int w_ind = 0;
        if (W_num == 1){
            w_ind = 0;
        } else if (W_num == E) {
            w_ind = e;
        } else if (W_num == N) {
            w_ind = i;
        }

        //const unsigned int j = edge_ij_e[E+e];
        const float r_ijx = rpos_ij_e[e*3+0];
        const float r_ijy = rpos_ij_e[e*3+1];
        const float r_ijz = rpos_ij_e[e*3+2];
        //const unsigned int ek = e*H+k;
        const float z = z_ek[e*H+k];

        #if VPE_DIM > 0
        #if CACHE_GV
        #pragma unroll
        for (int wdim = 0; wdim < V_HEAD_DIM; wdim++){
            shared_gv_ek[threadIdx.x][wdim] = gv_ekd[(e*H+k)*V_HEAD_DIM + wdim];
        }
        #endif

        // Compute backward of v' = Wv, as gW = (gv')^T * v
        #pragma unroll
        for (int dim = 0; dim < VPE_DIM; dim++)
            shared_gv[threadIdx.x][dim] = 0;

        #pragma unroll
        for (int wdim = 0; wdim < V_HEAD_DIM; wdim++){
            const float *W = &W_k[(w_ind*H+k)*V_HEAD_DIM*VPE_DIM + wdim*VPE_DIM];
            #if CACHE_GV
            float gv_val = shared_gv_ek[threadIdx.x][wdim];
            #else
            float gv_val = gv_ekd[(e*H+k)*V_HEAD_DIM + wdim];
            #endif

            #pragma unroll
            for (int dim = 0; dim < VPE_DIM; dim++)
                shared_gv[threadIdx.x][dim] += W[dim]*gv_val;
        }

        if (gW_k != NULL){
            #pragma unroll
            for (int dim = 0; dim < VPE_DIM; dim++)
                shared_v[threadIdx.x][dim] = 0;
        }
        #endif

        float px_avr = 0;
        float pbg_avr = 0;

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

            float p = expf(a*d2 - z);
            float px = d2*p;
            float d = sqrt(d2);
            px_avr += px;

            #if VPE_DIM > 0
            const float step = 1.0f/((float)wscale*sqrt(2.0f));
            const float interval = (float)dist_max/VPE_DIM;
            const float sig = interval * wscale;

            float bg = 0;

            #if VPE_DIM - LENGTH_RANGE > 8
            // CASE 1: ----------- Compute Gaussians within sigma*CUT_SIGMA ----------
            int start_dim = (d-CUT_SIGMA*sig)/interval - 1;
            start_dim = min(start_dim, VPE_DIM-LENGTH_RANGE);
            start_dim = max(0, start_dim);

            float b = -d/interval*step + step*start_dim;

            float *ptr_v = &shared_v[threadIdx.x][start_dim];
            float *ptr_gv = &shared_gv[threadIdx.x][start_dim];
            #pragma unroll
            for (int dim = 0; dim < LENGTH_RANGE; dim++)
            {
                b += step;
                float gauss = expf(-b*b);
                bg += gauss*ptr_gv[dim];
                ptr_v[dim] += gauss*p;
            }

            #else
            // Case 2: ------------ Compute all Gaussians (full range) -------

            float b = -d/interval*step;
            float *ptr_v = shared_v[threadIdx.x];
            float *ptr_gv = shared_gv[threadIdx.x];
            #pragma unroll
            for (int dim = 0; dim < VPE_DIM; dim++)
            {
                b += step;
                float gauss = expf(-b*b);
                bg += gauss*ptr_gv[dim];
                ptr_v[dim] += gauss*p;
            }
            #endif // VPE_DIM - LENGTH_RANGE <= 8
            #endif // VPE_DIM > 0
            sum_v += px*bg;
            pbg_avr += p*bg;
        }

        sum += px_avr*gz_ek[e*H+k];
        sum_v -= px_avr*pbg_avr;

        #pragma unroll
        for (int wdim = 0; wdim < V_HEAD_DIM; wdim++){
            #if CACHE_GV
            float gv_val = shared_gv_ek[threadIdx.x][wdim];
            #else
            float gv_val = gv_ekd[(e*H+k)*V_HEAD_DIM + wdim];
            #endif

            #pragma unroll
            for (int dim = 0; dim < VPE_DIM; dim++)
                gW_k[wdim*VPE_DIM+dim] += shared_v[threadIdx.x][dim]*gv_val;
        }
    }// for edge e
    ga_ik[tid] = sum + sum_v;
}
#include <math_constants.h>
extern "C" __global__

// whether cache 'gv_ekd' in shared memory or load from global memory each time
#if PE_THREAD_NUM*(ANGLE_ENC_DIM+1)*2 + PE_THREAD_NUM*(V_HEAD_DIM+1) <= 1024
#define CACHE_GV 1
#else
#define CACHE_GV 0
#endif

void real_enc_frame_proj_bwd_angle(
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
    const float angle_sigma,
    const float* W_ks,
    const float* frame_vecs,
    const unsigned int W_num,
    const float* rveclens_n,
    const float cutoff_radius,
    float* ga_ik,
    float* gW_ks)
    {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N*H*AXIS_NUM) return;
    // tid = (axis*N + i)*H + k
    // tid = axis*N*H + i*H + k
    const unsigned int axis = tid / (N*H);
    const unsigned int k = tid % H;
    const unsigned int i = (tid/H) % N;
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

    #if ANGLE_ENC_DIM > 0
    #if CACHE_GV
    __shared__ float shared_gv_ek[PE_THREAD_NUM][V_HEAD_DIM+1];
    #endif

    __shared__ float shared_gv[PE_THREAD_NUM][ANGLE_ENC_DIM+1];
    __shared__ float shared_v[PE_THREAD_NUM][ANGLE_ENC_DIM+1];

    if (gW_ks != NULL && (W_num == N || W_num == 1)){
        // gW_ks: (3, N, H, DH, DK)
        //gW_ks += ((i*H+k)*AXIS_NUM + axis)*V_HEAD_DIM*ANGLE_ENC_DIM;
        gW_ks += tid*V_HEAD_DIM*ANGLE_ENC_DIM;
        #pragma unroll
        for (int dim = 0; dim < V_HEAD_DIM*ANGLE_ENC_DIM; dim++){
            gW_ks[dim] = 0;
        }
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

        #if ANGLE_ENC_DIM > 0
        #if CACHE_GV
        #pragma unroll
        for (int wdim = 0; wdim < V_HEAD_DIM; wdim++){
            shared_gv_ek[threadIdx.x][wdim] = gv_ekd[(e*H+k)*V_HEAD_DIM + wdim];
        }
        #endif
        const float *frame = &frame_vecs[((axis*N+i)*H+k)*3];//&frame_vecs[(i*H+k)*3*AXIS_NUM + axis*3];
        float frame_x = frame[0];
        float frame_y = frame[1];
        float frame_z = frame[2];

        #pragma unroll
        for (int dim = 0; dim < ANGLE_ENC_DIM; dim++)
            shared_v[threadIdx.x][dim] = 0;

        #pragma unroll
        for (int dim = 0; dim < ANGLE_ENC_DIM; dim++) {
            shared_gv[threadIdx.x][dim] = 0;
        }

        #pragma unroll
        for (int wdim = 0; wdim < V_HEAD_DIM; wdim++){
            //const float *W = &W_ks[(w_ind*H+k)*V_HEAD_DIM*ANGLE_ENC_DIM*AXIS_NUM + V_HEAD_DIM*ANGLE_ENC_DIM*axis + wdim*ANGLE_ENC_DIM];
            //const float *W = W_ks + (w_ind*H+k)*V_HEAD_DIM*ANGLE_ENC_DIM*AXIS_NUM + V_HEAD_DIM*ANGLE_ENC_DIM*axis + wdim*ANGLE_ENC_DIM;
            const float *W = W_ks + ((axis*W_num+w_ind)*H + k)*V_HEAD_DIM*ANGLE_ENC_DIM + wdim*ANGLE_ENC_DIM;
            #if CACHE_GV
            float gv_val = shared_gv_ek[threadIdx.x][wdim];
            #else
            float gv_val = gv_ekd[(e*H+k)*V_HEAD_DIM + wdim];
            #endif

            #pragma unroll
            for (int dim = 0; dim < ANGLE_ENC_DIM; dim++)
                shared_gv[threadIdx.x][dim] += W[dim]*gv_val;
        }
        #endif // ANGLE_ENC_DIM

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
            float bg = 0;

            #if ANGLE_ENC_DIM > 0
            #if COS_ABS == 0
            float step = 2.0f/((ANGLE_ENC_DIM - 1.0f) * angle_sigma * sqrtf(2));
            float interval = 2.0f/(ANGLE_ENC_DIM - 1.0f);
            #else
            float step = 1.0f/((ANGLE_ENC_DIM - 1.0f) * angle_sigma * sqrtf(2));
            float interval = 1.0f/(ANGLE_ENC_DIM - 1.0f);
            #endif
            float sigma_inv = 1.0f / (angle_sigma * sqrtf(2));

            float cos_angle = 1e3;
            if (d2 > 1e-5){
                cos_angle = (dx*frame_x + dy*frame_y + dz*frame_z) / d;
            }
            #if COS_ABS
            cos_angle = fabsf(cos_angle);
            #endif

            #if ANGLE_ENC_DIM-ANGLE_RANGE > 8
            // CASE 1: ----------- Compute Gaussians within sigma*CUT_SIGMA ----------
            int end_dim = min((cos_angle + 1.0f + CUT_SIGMA*angle_sigma)/interval + 1,  (float)ANGLE_ENC_DIM);
            end_dim = max(end_dim, ANGLE_RANGE);
            float angle = (-cos_angle + 1.0f) * sigma_inv - (ANGLE_ENC_DIM - end_dim)*step;

            // NOTE:
            // It is important to access shared_v and shared_gv via
            // pointers created before the loop, instead of accessing
            // them directly as shared_v[threadIdx.x][end_dim-1-m].
            // In some environment, the direct accessing increases
            // the register usage when the loop is unrolled, making
            // the code much slower.
            float *ptr_v = &shared_v[threadIdx.x][end_dim-1];
            float *ptr_gv = &shared_gv[threadIdx.x][end_dim-1];
            #pragma unroll
            for (int m = 0; m < ANGLE_RANGE; m++)
            {
                float gauss = expf(-angle*angle);
                ptr_v[-m] += gauss*p;
                bg += gauss*ptr_gv[-m];
                angle -= step;
            }

            #else
            // Case 2: ------------ Compute all Gaussians (full range) -------
            float angle = (-cos_angle + 1.0f) * sigma_inv - (ANGLE_ENC_DIM-1)*step;
            // interval = 2/(D-1)
            // step = 2/((D-1)*sqrt(2) sigma) = interval*sigma_inv
            // sigma_inv = 1/(sqrt(2) sigma)
            // bk(x) = exp( -(x-k'*interval)^2/2 sigma^2 )
            //       = exp( -[(x-k'*interval)*sigma_inv]^2 )
            //       = exp( -[x*sigma_inv - k'*step]^2 )
            // x = 1 - cos_angle
            // k' = D-1-k

            float *ptr_v = shared_v[threadIdx.x];
            float *ptr_gv = shared_gv[threadIdx.x];
            #pragma unroll
            for (int dim = 0; dim < ANGLE_ENC_DIM; dim++)
            {
                float gauss = expf(-angle*angle);
                ptr_v[dim] += gauss*p;
                bg += gauss*ptr_gv[dim];
                angle += step;
            }
            #endif // ANGLE_ENC_DIM-ANGLE_RANGE <= 8
            #endif // ANGLE_ENC_DIM > 0

            sum_v += px*bg;
            pbg_avr += p*bg;
        }

        #if ANGLE_ENC_DIM > 0
        #pragma unroll
        for (int wdim = 0; wdim < V_HEAD_DIM; wdim++){
            #if CACHE_GV
            float gv_val = shared_gv_ek[threadIdx.x][wdim];
            #else
            float gv_val = gv_ekd[(e*H+k)*V_HEAD_DIM + wdim];
            #endif

            #pragma unroll
            for (int dim = 0; dim < ANGLE_ENC_DIM; dim++)
                gW_ks[wdim*ANGLE_ENC_DIM+dim] += shared_v[threadIdx.x][dim]*gv_val;
        }
        #endif

        sum += px_avr*gz_ek[e*H+k];
        sum_v -= px_avr*pbg_avr;

    }// for edge e
    
    //ga_ik[tid] = sum + sum_v;
    ga_ik[tid] = sum_v;
}
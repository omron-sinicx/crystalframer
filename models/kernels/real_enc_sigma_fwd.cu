#include <math_constants.h>
#include <curand_kernel.h>
extern "C" __global__


void real_enc_sigma_fwd(
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
    const long long int W_num,
    const float* rveclens_n,
    const double cutoff_radius,
    float* sigma_mat,
    curandState *state_buff,
    unsigned long long seed,
    int* edge_ij_e_select,
    float* sigma_mat_max){
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

    for (int dim = 0; dim < VPE_DIM; dim++)
        sv[dim] = 0;
    const float reci_ws_sqrt2 = 1.0f/((float)wscale*sqrt(2.0f));
    const float mu0 = (float)dist_max/VPE_DIM;
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
    float sum = 0;

    float sum_wo_0 = 0;
    float d2_max = 100000000;
    int atom = edge_ij_e[e];
    long long int idx = 1000 * tid;
    long long int divval = E * H * 1000;

    curandState *state = state_buff + tid;
    curand_init(seed, tid, 0, state);

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
        float w = expf(a*(d2 - d2min));

        sum += w;

        sigma_mat[tid * 6 + 0] += dx * dx * w;
        sigma_mat[tid * 6 + 1] += dx * dy * w;
        sigma_mat[tid * 6 + 2] += dx * dz * w;
        sigma_mat[tid * 6 + 3] += dy * dy * w;
        sigma_mat[tid * 6 + 4] += dy * dz * w;
        sigma_mat[tid * 6 + 5] += dz * dz * w;

        long long int div_result = idx % divval;

        float rand = (curand_uniform(state)-0.5)*SYMM_BREAK_NOISE;

        if (d2 > 1e-5 && d2_max > d2 + rand)
        {
        d2_max = d2+rand;
        sigma_mat_max[(atom*H+k) * 6 + 0] = dx * dx / d2;
        sigma_mat_max[(atom*H+k) * 6 + 1] = dx * dy / d2;
        sigma_mat_max[(atom*H+k) * 6 + 2] = dx * dz / d2;
        sigma_mat_max[(atom*H+k) * 6 + 3] = dy * dy / d2;
        sigma_mat_max[(atom*H+k) * 6 + 4] = dy * dz / d2;
        sigma_mat_max[(atom*H+k) * 6 + 5] = dz * dz / d2;
        }
        idx += 1;
    }

    for (int dim = 0; dim < 6; dim++){
        sigma_mat[tid * 6 + dim] = sigma_mat[tid * 6 + dim] / sum;
    }
}


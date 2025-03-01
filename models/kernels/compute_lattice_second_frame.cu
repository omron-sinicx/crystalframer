#include <math_constants.h>
#include <curand_kernel.h>
extern "C" __global__


void compute_lattice_second_frame(
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
    const float* rveclens_n,
    const double cutoff_radius,
    curandState *state_buff,
    unsigned long long seed,
    float* dx_first,
    float* dy_first,
    float* dz_first,
    float* dx_second,
    float* dy_second,
    float* dz_second
    ){
    const long long int tid = (long long int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= E*H) return;

    const long long int k = tid%H;
    const long long int e = tid/H;
    const long long int i = edge_ij_e[e];
    const long long int j = edge_ij_e[E+e];

    if (i != j) return;

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

    float d2_min = 1e10;
    long long int idx = 1000*tid;
    long long int max_idx = E*H*1000;
    float sum = 0;
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
        float d = sqrtf(d2);

        float rand = (curand_uniform(state)-0.5)*SYMM_BREAK_NOISE;
        
        if (d2_min > d2 + rand && d2 > 1e-5)
        {
        float dx_norm = dx/d;
        float dy_norm = dy/d;
        float dz_norm = dz/d;

        float cos_term = dx_norm * dx_first[i*H+k] + dy_norm * dy_first[i*H+k] + dz_norm * dz_first[i*H+k];

        if (1 - fabsf(cos_term) > 1e-5){
            if (cos_term<0){
            dx_second[i*H+k] = -dx_norm;
            dy_second[i*H+k] = -dy_norm;
            dz_second[i*H+k] = -dz_norm;
            } else {
            dx_second[i*H+k] = dx_norm;
            dy_second[i*H+k] = dy_norm;
            dz_second[i*H+k] = dz_norm;
            }
            d2_min = d2 + rand;
        }
        }
        idx += 1;
    }
}
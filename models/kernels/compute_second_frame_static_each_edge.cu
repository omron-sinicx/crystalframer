#include <math_constants.h>
#include <curand_kernel.h>
extern "C" __global__


void compute_second_frame_static_each_edge(
    const float* a_ik,
    const float* rpos_ij_e,
    const float* dist2_min_e,
    const float* tvecs_n,
    const long long int* batch_i,
    const long long int* edge_ij_e,
    const unsigned int N,
    const unsigned int H,
    const unsigned int E,
    const float* rveclens_n,
    const float cutoff_radius,
    curandState *state_buff,
    unsigned long long seed,
    float* dx_first,
    float* dy_first,
    float* dz_first,
    float* dx_second,
    float* dy_second,
    float* dz_second,
    float* second_value
    ){
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= E*H) return;


    const unsigned int k = tid%H;
    const unsigned int e = tid/H;
    const unsigned int i = edge_ij_e[e];
    const unsigned int j = edge_ij_e[E+e];
    const unsigned int n = batch_i[i];
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
            if (d2 > 1e-5){
            d2min = fminf(d2min, d2);
            }
        }
    } else {
        d2min = dist2_min_e[e];
    }

    float max_val = -1e15;
    curandState *state = state_buff + tid;
    curand_init(seed, tid, 0, state);

    float dx_second_, dy_second_, dz_second_, second_value_;
    float dx1 = dx_first[i*H + k];
    float dy1 = dy_first[i*H + k];
    float dz1 = dz_first[i*H + k];
    
    for (float n1 = -R1; n1 <= R1; n1++)
    for (float n2 = -R2; n2 <= R2; n2++)
    for (float n3 = -R3; n3 <= R3; n3++)
    {
        float dx = r_ijx + t1_x*n1 + t2_x*n2 + t3_x*n3;
        float dy = r_ijy + t1_y*n1 + t2_y*n2 + t3_y*n3;
        float dz = r_ijz + t1_z*n1 + t2_z*n2 + t3_z*n3;
        float d2 = dx*dx + dy*dy + dz*dz;
        float d = sqrtf(d2);

        if (d2 > 1e-5)
        {
            float dx_norm = dx / d;
            float dy_norm = dy / d;
            float dz_norm = dz / d;

            float cos_term = dx_norm * dx1 + dy_norm * dy1 + dz_norm * dz1;
            float d2_add_log = d2-d2min + logf(max(1 - fabsf(cos_term), 1e-5));
            
            float rand = (curand_uniform(state)-0.5)*SYMM_BREAK_NOISE;
            
            if (d2_add_log > max_val + rand && fabsf(cos_term) < 0.9999){
                max_val = d2_add_log + rand;
                dx_second_ = dx_norm;
                dy_second_ = dy_norm;
                dz_second_ = dz_norm;
                second_value_ = d2_add_log + d2min;
            }
        }
    }
    dx_second[tid] = dx_second_;
    dy_second[tid] = dy_second_;
    dz_second[tid] = dz_second_;
    second_value[tid] = second_value_;
}

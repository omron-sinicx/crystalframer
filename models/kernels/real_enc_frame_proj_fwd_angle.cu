#include <math_constants.h>
extern "C" __global__

void real_enc_frame_proj_fwd_angle(
    const float* a_ik,
    const float* rpos_ij_e,
    const float* dist2_min_ek,
    const float* tvecs_n,
    const long long int* batch_i,
    const long long int* edge_ij_e,
    const unsigned int N,
    const unsigned char H,
    const unsigned int E,
    const float angle_sigma,
    const float* W_ks,
    const float* frame_vecs,
    const unsigned int W_num,
    const float* rveclens_n,
    const float cutoff_radius,
    float* z_ek,
    float* v_ekd){
    const unsigned int tid = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= E*H*AXIS_NUM) return;
    // tid = (axis*E + e)*H + k
    // tid = axis*E*H + e*H + k

    const unsigned int axis = tid / (E*H);
    const unsigned int k = tid % H;
    const unsigned int e = (tid/H) % E;
    const unsigned int i = edge_ij_e[e];
    //const unsigned int j = edge_ij_e[E+e];
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
    v_ekd += tid*V_HEAD_DIM;

    // ------ for value pos enc  -----
    #if ANGLE_ENC_DIM > 0
    __shared__ float shared_v[PE_THREAD_NUM][ANGLE_ENC_DIM+1];
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

        #if MINIMUM_RANGE > 0
        R1 = max(R1, MINIMUM_RANGE);
        R2 = max(R2, MINIMUM_RANGE);
        R3 = max(R3, MINIMUM_RANGE);
        #endif
    }

    float d2min = 1e10;
    if (dist2_min_ek == NULL)
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
        d2min = dist2_min_ek[e*H+k];
    }


    #if ANGLE_ENC_DIM > 0
    #pragma unroll
    for (int dim = 0; dim < ANGLE_ENC_DIM; dim++)
        shared_v[threadIdx.x][dim] = 0;
    
    const float *frame = &frame_vecs[((axis*N+i)*H+k)*3];//&frame_vecs[(i*H+k)*3*AXIS_NUM + axis*3];
    float frame_x = frame[0];
    float frame_y = frame[1];
    float frame_z = frame[2];
    #endif

    float sum = 0;
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

        // ------ for angle-based value pos enc  -----
        #if ANGLE_ENC_DIM > 0
        #if COS_ABS == 0
        float step = 2.0f/((ANGLE_ENC_DIM - 1) * angle_sigma * sqrtf(2));
        float interval = 2.0f/(ANGLE_ENC_DIM - 1);
        #else
        float step = 1.0f/((ANGLE_ENC_DIM - 1) * angle_sigma * sqrtf(2));
        float interval = 1.0f/(ANGLE_ENC_DIM - 1);
        #endif
        float sigma_inv = 1.0f / (angle_sigma * sqrtf(2));

        float cos_angle = 1e3;
        if(d2 > 1e-5){
            //cos_angle = (dx*frame_3x3[threadIdx.x][axis][0] + dy*frame_3x3[threadIdx.x][axis][1] + dz*frame_3x3[threadIdx.x][axis][2]) / sqrtf(d2);
            cos_angle = (dx*frame_x + dy*frame_y + dz*frame_z) / sqrtf(d2);
        }
        #if COS_ABS
        cos_angle = fabsf(cos_angle);
        #endif

        #if ANGLE_ENC_DIM - ANGLE_RANGE > 8
        int end_dim = min((cos_angle + 1.0f + CUT_SIGMA*angle_sigma)/interval + 1.0f,  (float)ANGLE_ENC_DIM);
        end_dim = max(end_dim, ANGLE_RANGE);
        float angle = (-cos_angle + 1.0f) * sigma_inv - (ANGLE_ENC_DIM - end_dim)*step;
        
        #pragma unroll
        for (int dim = 0; dim < ANGLE_RANGE; dim++)
        {
            shared_v[threadIdx.x][end_dim-dim-1] += exp(-angle*angle)*w;
            angle -= step;
        }
        
        #else // ANGLE_ENC_DIM - ANGLE_RANGE <= 8
        float angle = (-cos_angle + 1.0f) * sigma_inv - (ANGLE_ENC_DIM - 1)*step;
        
        #pragma unroll
        for (int dim = 0; dim < ANGLE_ENC_DIM; dim++)
        {
            shared_v[threadIdx.x][dim] += exp(-angle*angle)*w;
            angle += step;
        }
        #endif // ANGLE_ENC_DIM - ANGLE_RANGE <= 8
        #endif // ANGLE_ENC_DIM > 0
        // ----
    }

    // ------ for angle-based value pos enc  -----
    // Do the matrix-vector multiplication: v' = Wv.
    #if ANGLE_ENC_DIM > 0
    unsigned int w_ind = 0;
    if (W_num == 1){
        w_ind = 0;
    } else if (W_num == E) {
        w_ind = e;
    } else if (W_num == N) {
        w_ind = i;
    }
    #pragma unroll
    for (int wdim = 0; wdim < V_HEAD_DIM; wdim++){
        float sum_v = 0;

        //const float *W = &W_ks[((w_ind*H+k)*V_HEAD_DIM*ANGLE_ENC_DIM)*AXIS_NUM + V_HEAD_DIM*ANGLE_ENC_DIM*axis+ wdim*ANGLE_ENC_DIM];
        const float *W = &W_ks[((axis*W_num+w_ind)*H + k)*V_HEAD_DIM*ANGLE_ENC_DIM + wdim*ANGLE_ENC_DIM];
        
        #pragma unroll
        for (int dim = 0; dim < ANGLE_ENC_DIM; dim++){
            sum_v += W[dim]*(shared_v[threadIdx.x][dim]);
        }

        v_ekd[wdim] = sum_v/sum;
    }
    #endif

    if (z_ek != NULL)
        z_ek[tid] = logf(sum) + d2min*a;
}
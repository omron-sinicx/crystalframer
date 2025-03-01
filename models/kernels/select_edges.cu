#include <math_constants.h>
#include <curand_kernel.h>
extern "C" __global__

void select_edges(
    const long long int* edge_ij_e,
    int* edge_ij_e_select,
    curandState *state_buff,
    unsigned long long seed,
    const long long int E,
    const long long int N,
    const int H){
    const long long int tid = (long long int)blockDim.x * blockIdx.x + threadIdx.x;
    int count = 0;
    int dif = 0;
    int atom = -1;

    curandState *state = state_buff + tid;
    curand_init(seed, tid, 0, state);

    for (int num = 0; num < E - 1; num++){
        count += 1;
        dif = edge_ij_e[num + 1] - edge_ij_e[num];
        if (dif != 0) {
            atom += 1;
            int random_index = (int)floor(curand_uniform(state)*count);
            edge_ij_e_select[(num-random_index)*H+tid] = 1;
            count = 0;
        }
    }
    count += 1;
    atom += 1;
    int random_index = (int)floor(curand_uniform(state)*count);
    edge_ij_e_select[(E-1-random_index)*H+tid] = 1;
}
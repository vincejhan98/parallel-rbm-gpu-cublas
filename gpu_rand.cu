#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#define NUM_THREADS 1024

// Put any static global variables here that you will use throughout the simulation.
int blks;
int bin_len;
double bin_size;

// arrays for coordination
int *part_ids;
int *bin_ids;
int *bin_ids_copy;

__global__ void sigmoid_and_sample(float *data, float* d_out, int num_nodes, curandState *d_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sigmoid;
    float rand;
    if (tid < num_nodes) {
        // data[tid] = 1 / (1 + exp(-data[tid]));
        sigmoid = 1 / (1 + exp(-data[tid]));
        rand = curand_uniform(&d_states[tid]);
        d_out[tid] = (sigmoid > rand)  ? 1 : 0;
    }
}

void simulate_one_step(int num_nodes, float* d_visibles, float* d_hiddens, float* d_weights, float* d_visible_bias, float* d_hidden_bias, float* d_tmp, float* d_rev_tmp, cublasHandle_t handle, curandState *d_states) {

    // =================================
    // 1. Visible to Hidden
    // =================================

    // copy bias values to d_tmp, to use cublasSgemv
    cudaMemcpy(d_tmp, d_hidden_bias, num_nodes * sizeof(float), cudaMemcpyDeviceToDevice);

    // Perform matrix-vector multiplication
    float alpha = 1.0f;
    float beta = 1.0f;

    // W @ V + h_bias
    cublasSgemv(handle, CUBLAS_OP_N, num_nodes, num_nodes, &alpha, d_weights, num_nodes, d_visibles, 1, &beta, d_tmp, 1);

    // Sigmoid & sample forward
    int block_size = 256;
    int num_blocks = (num_nodes + block_size - 1) / block_size;
    sigmoid_and_sample<<<num_blocks, block_size>>>(d_tmp, d_hiddens, num_nodes, d_states);

    cudaDeviceSynchronize();

    // =================================
    // 2. Hidden to Visible
    // =================================

    // copy bias values to d_tmp, to use cublasSgemv
    cudaMemcpy(d_tmp, d_visible_bias, num_nodes * sizeof(float), cudaMemcpyDeviceToDevice);

    // H @ W + h_bias
    cublasSgemv(handle, CUBLAS_OP_T, num_nodes, num_nodes, &alpha, d_weights, num_nodes, d_hiddens, 1, &beta, d_tmp, 1);

    // sigmoid and sample
    sigmoid_and_sample<<<num_blocks, block_size>>>(d_tmp, d_visibles, num_nodes, d_states);
    cudaDeviceSynchronize();
}

#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cstdlib>

// =================
// Helper Functions
// =================

// I/O routines
// save(fsave, visibles, num_nodes);
void save(std::ofstream& fsave, float* visibles, int num_nodes) {

    for (int i = 0; i < num_nodes; ++i) {
        fsave << visibles[i] << " ";
    }

    fsave << std::endl;
}

// Particle Initialization
void init_particles(particle_t* parts, int num_parts, double size, int part_seed) {
    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd());

    int sx = (int)ceil(sqrt((double)num_parts));
    int sy = (num_parts + sx - 1) / sx;

    std::vector<int> shuffle(num_parts);
    for (int i = 0; i < shuffle.size(); ++i) {
        shuffle[i] = i;
    }

    for (int i = 0; i < num_parts; ++i) {
        // Make sure particles are not spatially sorted
        std::uniform_int_distribution<int> rand_int(0, num_parts - i - 1);
        int j = rand_int(gen);
        int k = shuffle[j];
        shuffle[j] = shuffle[num_parts - i - 1];

        // Distribute particles evenly to ensure proper spacing
        parts[i].x = size * (1. + (k % sx)) / (1 + sx);
        parts[i].y = size * (1. + (k / sx)) / (1 + sy);

        // Assign random velocities within a bound
        std::uniform_real_distribution<float> rand_real(-1.0, 1.0);
        parts[i].vx = rand_real(gen);
        parts[i].vy = rand_real(gen);
    }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

void init_nodes(int num_nodes, int seed, float* nodes) {
    std::default_random_engine rng(seed); // Initialize Mersenne Twister random number generator with seed
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f); // Uniform distribution for float values between 0 and 1

    // Generate random float values for each node
    for (int i = 0; i < num_nodes; i++) {
        nodes[i] = std::round(distribution(rng)); // Generate a random float value between 0 and 1
    }
}

void init_weights_bias(int num_nodes, float* weights,
               float* visible_bias, float* hidden_bias) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0, 100.0); // Distribution for float values between -100 and 100

    // Initialize weights matrix and its transpose
    for (int i = 0; i < num_nodes * num_nodes; ++i) {
        weights[i] = dis(gen); // Random float value for weight
    }

    // Initialize visible bias
    for (int i = 0; i < num_nodes; ++i) {
        visible_bias[i] = dis(gen); // Random float value for visible bias
    }

    // Initialize hidden bias
    for (int i = 0; i < num_nodes; ++i) {
        hidden_bias[i] = dis(gen); // Random float value for hidden bias
    }
}


__global__ void initialize_curand_states(curandState *states, unsigned long long seed_offset, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long seed = seed_offset + tid;
    if (tid < num_nodes) {
        curand_init(seed, 0, 0, &states[tid]); // Initialize state with unique seed for each thread
    }
}


// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set number of nodes" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set node initialization seed" << std::endl;
        std::cout << "-i <int>: iterations" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    // Initialize Nodes
    int seed = find_int_arg(argc, argv, "-s", 0);
    int iterations = find_int_arg(argc, argv, "-i", 5000);

    // Initial nodes setup
    int num_nodes = find_int_arg(argc, argv, "-n", 3);
    float* visibles = new float[num_nodes];
    float* hiddens  = new float[num_nodes];
    init_nodes(num_nodes, seed, visibles);
    init_nodes(num_nodes, seed, hiddens);

    // Experiment 3
    
    float* weights   = new float[num_nodes * num_nodes];

    float* visible_bias = new float[num_nodes];
    float* hidden_bias  = new float[num_nodes];
    init_weights_bias(num_nodes, weights, visible_bias, hidden_bias);

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Initialize randomization variables
    float *d_random_numbers;
    curandState *d_states;
    cudaMalloc(&d_random_numbers, num_nodes * sizeof(float));
    cudaMalloc(&d_states, num_nodes * sizeof(curandState));
    int block_size = 256;
    int num_blocks = (num_nodes + block_size - 1) / block_size;
    initialize_curand_states<<<num_blocks, block_size>>>(d_states, seed, num_nodes);

    // Allocate device memory
    float *d_weights, *d_visibles, *d_hiddens, *d_visible_bias, *d_hidden_bias, *d_tmp, *d_rev_tmp;
    cudaMalloc(&d_weights, num_nodes * num_nodes * sizeof(float));
    cudaMalloc(&d_visibles, num_nodes * sizeof(float));
    cudaMalloc(&d_hiddens, num_nodes * sizeof(float));
    cudaMalloc(&d_visible_bias, num_nodes * sizeof(float));
    cudaMalloc(&d_hidden_bias, num_nodes * sizeof(float));
    cudaMalloc(&d_tmp, num_nodes * sizeof(float));
    cudaMalloc(&d_rev_tmp, num_nodes * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_weights, weights, num_nodes * num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visibles, visibles, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hiddens, hiddens, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visible_bias, visible_bias, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_bias, hidden_bias, num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    for (int step = 0; step < iterations; ++step) {
        simulate_one_step(num_nodes, d_visibles, d_hiddens, d_weights, d_visible_bias, d_hidden_bias, d_tmp, d_rev_tmp, handle, d_states);
        cudaDeviceSynchronize();
        if (fsave.good()) {
            cudaMemcpy(visibles, d_visibles, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
            save(fsave, visibles, num_nodes);
        }
    }

    // Copy result from device to host
    cudaMemcpy(hiddens, d_hiddens, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_weights);
    cudaFree(d_visibles);
    cudaFree(d_hiddens);
    cudaFree(d_visible_bias);
    cudaFree(d_hidden_bias);
    cudaFree(d_tmp);
    cudaFree(d_rev_tmp);

    cublasDestroy(handle);

    return 0;
}
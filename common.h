#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__
#include <cublas_v2.h>
#include <curand_kernel.h>

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005

// Particle Data Structure
typedef struct particle_t {
    double x;  // Position X
    double y;  // Position Y
    double vx; // Velocity X
    double vy; // Velocity Y
    double ax; // Acceleration X
    double ay; // Acceleration Y
} particle_t;

// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size);
void simulate_one_step(particle_t* parts, int num_parts, double size, double* times);
void simulate_one_step(particle_t* parts, int num_parts, double size);
void simulate_one_step(int num_nodes, float* d_visibles, float* d_hiddens, float* d_weights, float* d_visible_bias, float* d_hidden_bias, float* d_tmp, float* d_rev_tmp, cublasHandle_t handle, curandState *d_states);
#endif

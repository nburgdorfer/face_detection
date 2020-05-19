#ifndef _INTEGRALS_H_
#define _INTEGRALS_H_

#include "adaboost.h"

// kernels
__global__ void transpose_pitched(float *images, size_t pitch);
__global__ void transpose(struct Mat *images);
__global__ void parallel_scan_pitched(float *images, int rows, size_t pitch);
__global__ void parallel_scan_shared_mem_sb_pitched(float *images, int rows, size_t pitch);
__global__ void parallel_scan_shared_mem_db_picthed(float *images, int rows, size_t pitch);
__global__ void parallel_scan_shared_mem_db(struct Mat *images, int rows);

// main functions
void compute_integrals_d(struct Mat *images, int total_samples);
void compute_integrals_d_pitched(struct Mat *images, int total_samples);

#endif

#include "integrals.h"

__global__ void transpose_pitched(float *images, size_t pitch) {

    // get image for current block
    float *img = (float*) ((char*) images + blockIdx.x*pitch);

    float temp = img[(threadIdx.y*blockDim.x) + threadIdx.x];

    __syncthreads();

    img[(threadIdx.x*blockDim.x) + threadIdx.y] = temp;
}

__global__ void transpose(struct Mat *images) {

    float temp = images[blockIdx.x].values[(threadIdx.y*blockDim.x) + threadIdx.x];

    __syncthreads();

    images[blockIdx.x].values[(threadIdx.x*blockDim.x)+threadIdx.y] = temp;
}

__global__ void parallel_scan_pitched(float *images, int rows, size_t pitch) {

    float temp_val = 0.0;
    int offset = 0;
    int max_stride = ceil(blockDim.x/2.0);

    // get image for current block
    float *img = (float*) ((char*) images + blockIdx.x*pitch);

    // build image integral per block (576 threads) via Kogge-Stone Parallel Scan Algo (w/o double buffering)
    for (int stride=1; stride<=max_stride; stride*=2) {
        __syncthreads();
        offset = threadIdx.x - stride;
        if (offset >= 0) {
            temp_val = img[(threadIdx.y*rows)+threadIdx.x] + img[(threadIdx.y*rows)+offset];
        }

        __syncthreads();
        if (offset >= 0) {
            img[(threadIdx.y*rows)+threadIdx.x] = temp_val;
        }
    }
}

__global__ void parallel_scan_shared_mem_sb_pitched(float *images, int rows, size_t pitch) {

    // create shared memory array
    __shared__ float temp[576];

    float temp_val = 0.0;
    int offset = 0;
    int max_stride = ceil(blockDim.x/2.0);

    // get image for current block
    float *img = (float*) ((char*) images + blockIdx.x*pitch);

    // each thread pulls one pixel into shared
    temp[(threadIdx.y*rows)+threadIdx.x] = img[(threadIdx.y*rows)+threadIdx.x];

    // build image integral per block (576 threads) via Kogge-Stone Parallel Scan Algo (w/o double buffering)
    for (int stride=1; stride<=max_stride; stride*=2) {
        __syncthreads();
        offset = threadIdx.x - stride;
        if (offset >= 0) {
            temp_val = temp[(threadIdx.y*rows)+threadIdx.x] + temp[(threadIdx.y*rows)+offset];
        }

        __syncthreads();
        if (offset >= 0) {
            temp[(threadIdx.y*rows)+threadIdx.x] = temp_val;
        }
    }

    img[(threadIdx.y*rows)+threadIdx.x] = temp[(threadIdx.y*rows)+threadIdx.x];

}

__global__ void parallel_scan_shared_mem_db_pitched(float *images, int rows, size_t pitch) {
    // create shared memory arrays
    if (blockIdx.x==0 && threadIdx.x==0) {
        printf("test\n");
    }
    __shared__ float temp0[576];
    __shared__ float temp1[576];

    // get image for current block
    float *img = (float*) ((char*) images + blockIdx.x*pitch);
    if (blockIdx.x==0 && threadIdx.x==0) {
        printf("%f\n",img[0]);
    }

    // create pointers to shared memory arrays for double buffering
    float *source = temp0;
    float *dest = temp1;
    float *swap;

    int temp_val;
    float part_sum;
    int offset = 0;
    int max_stride = ceil(blockDim.x/2.0);

    // each thread pulls one pixel into shared
    temp_val = img[(threadIdx.y*rows)+threadIdx.x];
    temp0[(threadIdx.y*rows)+threadIdx.x] = temp_val;
    temp1[(threadIdx.y*rows)+threadIdx.x] = temp_val;

    // build image integral per block (576 threads) via Kogge-Stone Parallel Scan Algo (w/ double buffering)
    for (int stride=1; stride<=max_stride; stride*=2) {
        __syncthreads();
        offset = threadIdx.x - stride;
        part_sum = source[(threadIdx.y*rows)+threadIdx.x];

        if (offset >= 0) {
            part_sum += source[(threadIdx.y*rows)+offset];
        }

        dest[(threadIdx.y*rows)+threadIdx.x] = part_sum;

        swap = dest;
        dest = source;
        source = swap;
    }

    img[(threadIdx.y*rows)+threadIdx.x] = source[(threadIdx.y*rows)+threadIdx.x];
}


__global__ void parallel_scan_shared_mem_db(struct Mat *images, int rows) {
    // create shared memory arrays
    __shared__ float temp0[576];
    __shared__ float temp1[576];

    // create pointers to shared memory arrays for double buffering
    float *source = temp0;
    float *dest = temp1;
    float *swap;

    int temp_val;
    float part_sum;
    int offset = 0;
    int max_stride = ceil(blockDim.x/2.0);

    // each thread pulls one pixel into shared
    temp_val = images[blockIdx.x].values[(threadIdx.y*rows)+threadIdx.x];
    temp0[(threadIdx.y*rows)+threadIdx.x] = temp_val;
    temp1[(threadIdx.y*rows)+threadIdx.x] = temp_val;

    // build image integral per block (576 threads) via Kogge-Stone Parallel Scan Algo (w/ double buffering)
    for (int stride=1; stride<=max_stride; stride*=2) {
        __syncthreads();
        offset = threadIdx.x - stride;
        part_sum = source[(threadIdx.y*rows)+threadIdx.x];

        if (offset >= 0) {
            part_sum += source[(threadIdx.y*rows)+offset];
        }

        dest[(threadIdx.y*rows)+threadIdx.x] = part_sum;

        swap = dest;
        dest = source;
        source = swap;
    }

    images[blockIdx.x].values[(threadIdx.y*rows)+threadIdx.x] = source[(threadIdx.y*rows)+threadIdx.x];
}

void compute_integrals_d(struct Mat *images, int total_samples) {
    int rows = images[0].rows;
    int cols = images[0].cols;

    cudaError err;
    dim3 grid_size(total_samples,1);
    dim3 block_size(rows,cols);
    
    int img_bytes = sizeof(struct Mat) * total_samples;
    struct Mat *images_d;
    struct Mat *images_h = (struct Mat*) malloc(img_bytes);
    int val_bytes = sizeof(float) * rows*cols;

    for (int i=0; i<total_samples; ++i) {
        float *values;

        cudaMalloc(&values,val_bytes);
        cudaMemcpy(values,images[i].values,val_bytes,cudaMemcpyHostToDevice);

        images_h[i].rows = images[i].rows;
        images_h[i].cols = images[i].cols;
        images_h[i].values = values;
    }

    cudaMalloc(&images_d,img_bytes);
    cudaMemcpy(images_d,images_h,img_bytes,cudaMemcpyHostToDevice);
    
    /*** compute row sum ***/
    parallel_scan_shared_mem_db<<< grid_size,block_size >>>(images_d,rows);

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of kernel 'parallel_scan_shared_mem_db' #1: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*** take transpose ***/
    transpose<<< grid_size,block_size >>>(images_d);

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of kernel 'transpose' #1: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*** compute column sum ***/
    parallel_scan_shared_mem_db<<< grid_size,block_size >>>(images_d,rows);

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of kernel 'parallel_scan_shared_mem_db' #2: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*** take transpose ***/
    transpose<<< grid_size,block_size >>>(images_d);

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of kernel 'transpose' #2: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i=0; i<total_samples; ++i) {
        cudaMemcpy(images[i].values,images_h[i].values,val_bytes,cudaMemcpyDeviceToHost);
        cudaFree(images_h[i].values);
    }

    cudaFree(images_d);
}

void compute_integrals_d_pitched(struct Mat *images, int total_samples) {
    int rows = images[0].rows;
    int cols = images[0].cols;

    size_t dpitch;
    size_t hpitch = rows*cols*sizeof(float);
    int width = (rows*cols) * sizeof(float);
    int height = total_samples;

    cudaError err;
    dim3 grid_size(total_samples,1);
    dim3 block_size(rows,cols);
    
    
    /*** compute row sum ***/
    float *images_d;

    float **images_h = (float**) malloc(sizeof(float*) * total_samples);

    for (int i=0; i<total_samples; ++i) {
        images_h[i] = (float*) malloc(sizeof(float) * rows*cols);
        for (int j=0; j<rows*cols; ++j) {
            images_h[i][j] = images[i].values[j];
        }
    }

    cudaMallocPitch(&images_d, &dpitch, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemcpy2D(images_d, dpitch, images_h, hpitch, width, height, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    parallel_scan_shared_mem_db_pitched<<< grid_size,block_size >>>(images_d,rows,dpitch);

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of kernel 'parallel_scan_shared_mem_db' #1: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    /*** take transpose ***/
    transpose_pitched<<< grid_size,block_size >>>(images_d,dpitch);

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of kernel 'transpose' #1: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    /*** compute column sum ***/
    parallel_scan_shared_mem_db_pitched<<< grid_size,block_size >>>(images_d,rows,dpitch);

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of kernel 'parallel_scan_shared_mem_db' #2: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    /*** take transpose ***/
    transpose_pitched<<< grid_size,block_size >>>(images_d,dpitch);

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of kernel 'transpose' #2: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy2D(images_h, hpitch, images_d, dpitch, width, height, cudaMemcpyDeviceToHost);

    for (int i=0; i<total_samples; ++i) {
        for (int j=0; j<rows*cols; ++j) {
            images[i].values[j] = images_h[i][j];
        }
    }

    cudaFree(images_d);
}

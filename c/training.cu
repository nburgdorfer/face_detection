#include "training.h"

__constant__ float weights_d[MAX_SAMPLES];

__global__ void train(float *res, struct weak_classifier *weak_classifiers, float *images, int rows, size_t pitch, int total_samples, int total_features) {
    int sample_ind = ((blockIdx.x*blockDim.x) + threadIdx.x) % total_samples;
    int feat_ind = floorf(((blockIdx.x*blockDim.x) + threadIdx.x) / (float)total_samples);

    if (feat_ind < total_features) {
        struct feature feat = weak_classifiers[feat_ind].feat;
        float *img = (float*) ((char*) images + sample_ind*pitch);

        int A,B,C,D;
        int num_sections = feat.num_sections;
        struct section sect;
        float sum = 0.0;

        for (int s=0; s<num_sections; ++s) {
            sect = feat.sections[s];
            A = ((sect.row*rows) + sect.col) - (rows + 1);
            B = ((sect.row*rows) + (sect.col+sect.width-1)) - rows;
            C = (((sect.row+sect.height-1)*rows) + sect.col) - 1;
            D = (((sect.row+sect.height-1)*rows) + (sect.col+sect.width-1));

            sum += sect.sign * (img[D] - img[B] - img[C] + img[A]);

        }

        res[(feat_ind*total_samples*2) + (sample_ind * 2)] = sum;
        res[(feat_ind*total_samples*2) + (sample_ind * 2) + 1] = weights_d[sample_ind];
    }
}

__global__ void train_shared(float *res, struct weak_classifier *weak_classifiers, float *images, int rows, size_t pitch, int total_samples, int total_features, int iteration) {
    int sample_ind = (iteration*blockDim.x) + threadIdx.x;
    int feat_ind = blockIdx.x;

    if (sample_ind < total_samples) {

        __shared__ struct weak_classifier wc;

        if(threadIdx.x==0) {
            wc = weak_classifiers[feat_ind];
        }
        __syncthreads();

        float *img = (float*) ((char*) images + sample_ind*pitch);

        int A,B,C,D;
        int num_sections = wc.feat.num_sections;
        struct section sect;
        float sum = 0.0;

        for (int s=0; s<num_sections; ++s) {
            sect = wc.feat.sections[s];
            A = ((sect.row*rows) + sect.col) - (rows + 1);
            B = ((sect.row*rows) + (sect.col+sect.width-1)) - rows;
            C = (((sect.row+sect.height-1)*rows) + sect.col) - 1;
            D = (((sect.row+sect.height-1)*rows) + (sect.col+sect.width-1));

            sum += sect.sign * (img[D] - img[B] - img[C] + img[A]);

        }

        res[(feat_ind*total_samples*2) + (sample_ind * 2)] = sum;
        res[(feat_ind*total_samples*2) + (sample_ind * 2) + 1] = weights_d[sample_ind];
    }
}

__global__ void classify(float *errors, int *results, struct weak_classifier *weak_classifiers, 
        float *images, int *labels, int rows, size_t pitch, int total_samples, int total_features) {
    int sample_ind = ((blockIdx.x*blockDim.x) + threadIdx.x) % total_samples;
    int feat_ind = floorf(((blockIdx.x*blockDim.x) + threadIdx.x) / (float)total_samples);

    if (feat_ind < total_features) {
        weak_classifier wc = weak_classifiers[feat_ind];
        float *img = (float*) ((char*) images + sample_ind*pitch);

        int A,B,C,D;
        int num_sections = wc.feat.num_sections;
        struct section sect;
        int label;
        int res;
        float sum = 0.0;

        for (int s=0; s<num_sections; ++s) {
            sect = wc.feat.sections[s];
            A = ((sect.row*rows) + sect.col) - (rows + 1);
            B = ((sect.row*rows) + (sect.col+sect.width-1)) - rows;
            C = (((sect.row+sect.height-1)*rows) + sect.col) - 1;
            D = (((sect.row+sect.height-1)*rows) + (sect.col+sect.width-1));

            sum += sect.sign * (img[D] - img[B] - img[C] + img[A]);
        }

        label = (sum * wc.parity) < (wc.thresh * wc.parity);

        res = abs(label-(labels[sample_ind]));
        results[(feat_ind*total_samples) + sample_ind] = res;
        errors[(feat_ind*total_samples) + sample_ind] = (weights_d[sample_ind] * res);
    }
}

__global__ void classify_shared(float *errors, int *results, struct weak_classifier *weak_classifiers, 
        float *images, int *labels, int rows, size_t pitch, int total_samples, int total_features, int iteration) {
    int sample_ind = (iteration*blockDim.x) + threadIdx.x;
    int feat_ind = blockIdx.x;

    if (sample_ind < total_samples) {

        __shared__ struct weak_classifier wc;

        if(threadIdx.x==0) {
            wc = weak_classifiers[feat_ind];
        }
        __syncthreads();

        float *img = (float*) ((char*) images + sample_ind*pitch);

        int A,B,C,D;
        int label;
        int res;
        int num_sections = wc.feat.num_sections;
        struct section sect;
        float sum = 0.0;

        for (int s=0; s<num_sections; ++s) {
            sect = wc.feat.sections[s];
            A = ((sect.row*rows) + sect.col) - (rows + 1);
            B = ((sect.row*rows) + (sect.col+sect.width-1)) - rows;
            C = (((sect.row+sect.height-1)*rows) + sect.col) - 1;
            D = (((sect.row+sect.height-1)*rows) + (sect.col+sect.width-1));

            sum += sect.sign * (img[D] - img[B] - img[C] + img[A]);

        }

        label = (sum * wc.parity) < (wc.thresh * wc.parity);

        res = abs(label-(labels[sample_ind]));
        results[(feat_ind*total_samples) + sample_ind] = res;
        errors[(feat_ind*total_samples) + sample_ind] = (weights_d[sample_ind] * res);
    }
}

float* train_weak_d(int **results, int total_features, struct weak_classifier *weak_classifiers, float *weights, struct data_t data, float neg_sum, float pos_sum) {

    int block_size;
    int grid_size;
    int width;
    int height;
    int num_sects;
    int sect_bytes;
    size_t lab_bytes;
    size_t wt_bytes;
    size_t wc_bytes;
    size_t res_bytes;
    size_t err_bytes;
    size_t dpitch;
    size_t hpitch;
    float *images_d;
    int *labels_d;
    float *res_d;
    float *res_h;
    float *errors_d;
    float *errors_h;
    int *results_d;
    int *results_h;
    struct weak_classifier *wcs_d;
    struct weak_classifier *wcs_h;
    cudaError err;

    int rows = data.images[0].rows;
    int cols = data.images[0].cols;
    float cm_neg_sum = 0.0;
    float cm_pos_sum = 0.0;
    int weight_len = data.total_samples;
    double total_msecs = 0.0;
    struct timeval stop, start;

    int *thresh = (int*) malloc(sizeof(int) * total_features);
    int *parity = (int*) malloc(sizeof(int) * total_features);
    float **images_h = (float**) malloc(sizeof(float*) * data.total_samples);
    float *min_err = (float*) malloc(sizeof(float)*total_features);
    float *neg = (float*) malloc(sizeof(float) * total_features);
    float *pos = (float*) malloc(sizeof(float) * total_features);
    float *errors = (float*) malloc(sizeof(float) * total_features);

    float **cm_pos = (float**) malloc(sizeof(float*) * total_features);
    float **cm_neg = (float**) malloc(sizeof(float*) * total_features);

    float ***res = (float***) malloc(sizeof(float**) * total_features);


    for (int i=0; i<total_features; ++i) {
        res[i] = (float**) malloc(sizeof(float*) * weight_len);
        images_h[i] = (float*) malloc(sizeof(float) * rows*cols);
        cm_pos[i] = (float*) malloc(sizeof(float) * weight_len);
        cm_neg[i] = (float*) malloc(sizeof(float) * weight_len);
        min_err[i] = -1;

        for(int j=0; j<weight_len; ++j) {
            res[i][j] = (float*) malloc(sizeof(float) * 2);
        }
    }


    printf("\tTraining weak classifiers...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/

    block_size = 1024;
    grid_size = (int) ceil((data.total_samples * total_features) / (float)block_size);

    // set kernel variables
    wc_bytes = sizeof(struct weak_classifier) * total_features;
    res_bytes = sizeof(float) * total_features*data.total_samples*2;
    hpitch = rows*cols*sizeof(float);
    width = (rows*cols) * sizeof(float);
    height = data.total_samples;

    printf("malloc res_h\n");
    // allocate space for host results
    if ((res_h = (float*) malloc(res_bytes)) == NULL) {
        printf("res_h: failed to allocate %lu bytes\n", res_bytes);
    }

    // "flatten" images into 2D array
    for (int i=0; i<data.total_samples; ++i) {
        for (int j=0; j<rows*cols; ++j) {
            images_h[i][j] = data.images[i].values[j];
        }
    }

    // allocate device data
    cudaMallocPitch(&images_d, &dpitch, width, height);
    cudaMalloc(&wcs_d, wc_bytes);
    cudaMalloc(&res_d, res_bytes);

    // copy over device data
    cudaMemcpy2D(images_d, dpitch, images_h, hpitch, width, height, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(weights_d, weights, sizeof(float)*data.total_samples, 0, cudaMemcpyHostToDevice);

    // allocate space for section arrays of features
    wcs_h = (struct weak_classifier*) malloc(sizeof(struct weak_classifier) * total_features);

    for (int i=0; i<total_features; ++i) {
        num_sects = weak_classifiers[i].feat.num_sections;
        sect_bytes = sizeof(struct section) * num_sects;
        struct section *sects;

        cudaMalloc(&sects,sect_bytes);
        cudaMemcpy(sects,weak_classifiers[i].feat.sections, sect_bytes, cudaMemcpyHostToDevice);
        wcs_h[i].parity = weak_classifiers[i].parity;
        wcs_h[i].thresh = weak_classifiers[i].thresh;
        wcs_h[i].feat.num_sections = weak_classifiers[i].feat.num_sections;
        wcs_h[i].feat.sections = sects;
    }
    cudaMemcpy(wcs_d, wcs_h, wc_bytes, cudaMemcpyHostToDevice);

    // launch kernel
    train<<<grid_size,block_size>>>(res_d, wcs_d, images_d, rows, dpitch, data.total_samples, total_features);

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of the kernel: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // copy results to host
    cudaMemcpy(res_h, res_d, res_bytes, cudaMemcpyDeviceToHost);

    // convert 1D res_h to 3D res
    for (int i=0; i<total_features; ++i) {
        for (int j=0; j<weight_len; ++j) {
            res[i][j][0] = res_h[(i*weight_len*2) + (j*2)];
            res[i][j][1] = res_h[(i*weight_len*2) + (j*2) + 1];
        }
    }

    // free device data
    cudaFree(res_d);
    free(res_h);

    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/



    printf("\tSorting (results,weights) vector...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/

    //sort weights
    for (int k=0; k<total_features; ++k) {
        qsort((void*)res[k],weight_len,sizeof(float*),sort_pair);
    }

    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/



    printf("\tBuilding cumulative weights...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/

    for (int k=0; k<total_features; ++k) {
        cm_neg_sum = 0.0;
        cm_pos_sum = 0.0;
        for (int i=0; i<weight_len; ++i) {
            if(data.labels[i] == 0) {
                cm_neg_sum += res[k][i][1];
                cm_neg[k][i] = cm_neg_sum;
                cm_pos[k][i] = cm_pos_sum;
            } else {
                cm_pos_sum += res[k][i][1];
                cm_neg[k][i] = cm_neg_sum;
                cm_pos[k][i] = cm_pos_sum;
            }
        }
    }
    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/




    printf("\tSetting weak classifier threshold and parity...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/

    for (int k=0; k<total_features; ++k) {
        for (int i=0; i<weight_len; ++i) {

            neg[k] = cm_pos[k][i] + (neg_sum - cm_neg[k][i]);
            pos[k] = cm_neg[k][i] + (pos_sum - cm_pos[k][i]);

            if (neg[k] < pos[k]) {
                if (neg[k] < min_err[k] || min_err[k] < 0) {
                    min_err[k] = neg[k];
                    parity[k] = -1;
                    thresh[k] = res[k][i][0];
                }
            } else {
                if (pos[k] < min_err[k] || min_err[k] < 0) {
                    min_err[k] = pos[k];
                    parity[k] = 1;
                    thresh[k] = res[k][i][0];
                }
            }
        }

        weak_classifiers[k].parity = parity[k];
        weak_classifiers[k].thresh = thresh[k];

    }
    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/


    // free all data
    printf("freeing data...\n");
    free(thresh);
    free(parity);
    free(min_err);
    free(neg);
    free(pos);
    for (int i=0; i<total_features; ++i) {
/*
        for(int j=0; j<weight_len; ++j) {
            free(res[i][j]);
        }
        free(res[i]);
*/
        free(images_h[i]);
        free(cm_pos[i]);
        free(cm_neg[i]);
    }

    free(cm_pos);
    free(cm_neg);
    free(images_h);
    free(res);


    printf("\tClassifying samples...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/

    block_size = 1024;
    grid_size = (int) ceil((data.total_samples * total_features) / (float)block_size);

    // set kernel variables
    lab_bytes = sizeof(int) * data.total_samples;
    wt_bytes = sizeof(float) * data.total_samples;
    wc_bytes = sizeof(struct weak_classifier) * total_features;
    err_bytes = sizeof(float) * total_features*data.total_samples;
    res_bytes = sizeof(int) * total_features*data.total_samples;
    hpitch = rows*cols*sizeof(float);
    width = (rows*cols) * sizeof(float);
    height = data.total_samples;

    // allocate host data
    results_h = (int*) malloc(sizeof(int) * total_features*data.total_samples);
    errors_h = (float*) malloc(sizeof(float) * total_features*data.total_samples);

    // allocate device data
    cudaMallocPitch(&images_d, &dpitch, width, height);
    cudaMalloc(&wcs_d, wc_bytes);
    cudaMalloc(&labels_d, lab_bytes);
    cudaMalloc(&results_d, res_bytes);

    // copy over device data

    cudaMemcpy2D(images_d, dpitch, images_h, hpitch, width, height, cudaMemcpyHostToDevice);
    cudaMemcpy(labels_d,data.labels,lab_bytes,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(weights_d, weights, wt_bytes, 0, cudaMemcpyHostToDevice);

    // allocate space for section arrays of features
    wcs_h = (struct weak_classifier*) malloc(sizeof(struct weak_classifier) * total_features);

    for (int i=0; i<total_features; ++i) {
        num_sects = weak_classifiers[i].feat.num_sections;
        sect_bytes = sizeof(struct section) * num_sects;
        struct section *sects;

        cudaMalloc(&sects,sect_bytes);
        cudaMemcpy(sects,weak_classifiers[i].feat.sections, sect_bytes, cudaMemcpyHostToDevice);
        wcs_h[i].parity = weak_classifiers[i].parity;
        wcs_h[i].thresh = weak_classifiers[i].thresh;
        wcs_h[i].feat.num_sections = weak_classifiers[i].feat.num_sections;
        wcs_h[i].feat.sections = sects;
    }
    cudaMemcpy(wcs_d, wcs_h, wc_bytes, cudaMemcpyHostToDevice);

    err_bytes = sizeof(float) * total_features*data.total_samples;
    cudaMalloc(&errors_d, err_bytes);

    // launch kernel
    classify<<<grid_size,block_size>>>(errors_d, results_d, wcs_d, images_d, labels_d, rows, dpitch, data.total_samples, total_features);

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of the kernel: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // copy errors and results to host
    cudaMemcpy(errors_h, errors_d, err_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(results_h, results_d, res_bytes, cudaMemcpyDeviceToHost);


    for (int i=0; i<total_features; ++i) {
        errors[i] = 0.0;
        for (int j=0; j<data.total_samples; ++j) {
            results[i][j] = results_h[(i*data.total_samples) + j];
            errors[i] += errors_h[(i*data.total_samples) + j];
        }
    }

    // free all device data
    for (int i=0; i<total_features; ++i) {
        cudaFree(wcs_h[i].feat.sections);
    }
    cudaFree(wcs_d);
    cudaFree(errors_d);
    cudaFree(images_d);
    free(wcs_h);

    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/



    return errors;
}

float* train_weak_d_shared(int **results, int total_features, struct weak_classifier *weak_classifiers, float *weights, struct data_t data, float neg_sum, float pos_sum) {

    int block_size;
    int grid_size;
    int width;
    int height;
    int num_sects;
    int sect_bytes;
    int total_iters;
    size_t lab_bytes;
    size_t wt_bytes;
    size_t wc_bytes;
    size_t res_bytes;
    size_t err_bytes;
    size_t dpitch;
    size_t hpitch;
    float *images_d;
    int *labels_d;
    float *res_d;
    //float *res_h;
    float *errors_d;
    float *errors_h;
    int *results_d;
    int *results_h;
    struct weak_classifier *wcs_d;
    struct weak_classifier *wcs_h;
    cudaError err;

    int rows = data.images[0].rows;
    int cols = data.images[0].cols;
    float cm_neg_sum = 0.0;
    float cm_pos_sum = 0.0;
    int weight_len = data.total_samples;
    double total_msecs = 0.0;
    struct timeval stop, start;

    int *thresh = (int*) malloc(sizeof(int) * total_features);
    int *parity = (int*) malloc(sizeof(int) * total_features);
    float **images_h = (float**) malloc(sizeof(float*) * data.total_samples);
    float **res_h = (float**) malloc(sizeof(float*) * total_features);
    float *min_err = (float*) malloc(sizeof(float)*total_features);
    float *neg = (float*) malloc(sizeof(float) * total_features);
    float *pos = (float*) malloc(sizeof(float) * total_features);
    float *errors = (float*) malloc(sizeof(float) * total_features);


    float ***res = (float***) malloc(sizeof(float**) * total_features);


    for (int i=0; i<total_features; ++i) {
        res[i] = (float**) malloc(sizeof(float*) * weight_len);

        for(int j=0; j<weight_len; ++j) {
            res[i][j] = (float*) malloc(sizeof(float) * 2);
        }
    }

    for (int i=0; i<total_features; ++i) {
        images_h[i] = (float*) malloc(sizeof(float) * rows*cols);
    }


    printf("\tTraining weak classifiers...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/

    block_size = 1024;
    grid_size = total_features;

    // set kernel variables
    wc_bytes = sizeof(struct weak_classifier) * total_features;
    res_bytes = sizeof(float) * total_features*data.total_samples*2;
    hpitch = rows*cols*sizeof(float);
    width = (rows*cols) * sizeof(float);
    height = data.total_samples;

    size_t res_dpitch;
    size_t res_hpitch = data.total_samples*2;
    int r_width = data.total_samples*2;
    int r_height = total_features;


    // allocate space for host results
    /*
    if (!(res_h = (float*) malloc(res_bytes))) {
        printf("res_h: failed to allocate %lu bytes\n", res_bytes);
    }
    */
    for (int i=0; i<total_features; ++i) {
        res_h[i] = (float*) malloc(sizeof(float) * data.total_samples * 2);
    }

    // "flatten" images into 2D array
    for (int i=0; i<data.total_samples; ++i) {
        for (int j=0; j<rows*cols; ++j) {
            images_h[i][j] = data.images[i].values[j];
        }
    }

    // allocate device data
    cudaMallocPitch(&images_d, &dpitch, width, height);
    cudaMalloc(&wcs_d, wc_bytes);
    //cudaMalloc(&res_d, res_bytes);
    cudaMallocPitch(&res_d, &res_dpitch, r_width, r_height);

    // copy over device data
    cudaMemcpy2D(images_d, dpitch, images_h, hpitch, width, height, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(weights_d, weights, sizeof(float)*data.total_samples, 0, cudaMemcpyHostToDevice);

    // allocate space for section arrays of features
    wcs_h = (struct weak_classifier*) malloc(sizeof(struct weak_classifier) * total_features);

    for (int i=0; i<total_features; ++i) {
        num_sects = weak_classifiers[i].feat.num_sections;
        sect_bytes = sizeof(struct section) * num_sects;
        struct section *sects;

        cudaMalloc(&sects,sect_bytes);
        cudaMemcpy(sects,weak_classifiers[i].feat.sections, sect_bytes, cudaMemcpyHostToDevice);
        wcs_h[i].parity = weak_classifiers[i].parity;
        wcs_h[i].thresh = weak_classifiers[i].thresh;
        wcs_h[i].feat.num_sections = weak_classifiers[i].feat.num_sections;
        wcs_h[i].feat.sections = sects;
    }
    cudaMemcpy(wcs_d, wcs_h, wc_bytes, cudaMemcpyHostToDevice);

    // launch kernel
    total_iters = (int) ceil(data.total_samples/(float)block_size);
    for (int i=0; i<total_iters; ++i) {
        train_shared<<<grid_size,block_size>>>(res_d, wcs_d, images_d, rows, dpitch, data.total_samples, total_features, i);
    }

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of the kernel: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // copy results to host
    //cudaMemcpy(res_h, res_d, res_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(res_d, res_dpitch, res_h, res_hpitch, r_width, r_height, cudaMemcpyHostToDevice);

    // convert 2D res_h to 3D res
    for (int i=0; i<total_features; ++i) {
        for (int j=0; j<weight_len; ++j) {
            res[i][j][0] = res_h[i][j*2];
            res[i][j][1] = res_h[i][(j*2)+1];
        }
    }

    // free device data
    cudaFree(res_d);

    for (int i=0; i<total_features; ++i) {
        free(res_h[i]);
    }
    free(res_h);

    printf("\tfreeing images_h...\n");
    for (int i=0; i<total_features; ++i) {
        free(images_h[i]);
    }
    free(images_h);

    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/



    printf("\tSorting (results,weights) vector...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/

    //sort weights
    for (int k=0; k<total_features; ++k) {
        qsort((void*)res[k],weight_len,sizeof(float*),sort_pair);
    }

    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/


    float **cm_pos = (float**) malloc(sizeof(float*) * total_features);
    float **cm_neg = (float**) malloc(sizeof(float*) * total_features);
    for (int i=0; i<total_features; ++i) {
        cm_pos[i] = (float*) malloc(sizeof(float) * weight_len);
        cm_neg[i] = (float*) malloc(sizeof(float) * weight_len);
        min_err[i] = -1;
    }

    printf("\tBuilding cumulative weights...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/

    for (int k=0; k<total_features; ++k) {
        cm_neg_sum = 0.0;
        cm_pos_sum = 0.0;
        for (int i=0; i<weight_len; ++i) {
            if(data.labels[i] == 0) {
                cm_neg_sum += res[k][i][1];
                cm_neg[k][i] = cm_neg_sum;
                cm_pos[k][i] = cm_pos_sum;
            } else {
                cm_pos_sum += res[k][i][1];
                cm_neg[k][i] = cm_neg_sum;
                cm_pos[k][i] = cm_pos_sum;
            }
        }
    }
    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/




    printf("\tSetting weak classifier threshold and parity...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/

    for (int k=0; k<total_features; ++k) {
        for (int i=0; i<weight_len; ++i) {

            neg[k] = cm_pos[k][i] + (neg_sum - cm_neg[k][i]);
            pos[k] = cm_neg[k][i] + (pos_sum - cm_pos[k][i]);

            if (neg[k] < pos[k]) {
                if (neg[k] < min_err[k] || min_err[k] < 0) {
                    min_err[k] = neg[k];
                    parity[k] = -1;
                    thresh[k] = res[k][i][0];
                }
            } else {
                if (pos[k] < min_err[k] || min_err[k] < 0) {
                    min_err[k] = pos[k];
                    parity[k] = 1;
                    thresh[k] = res[k][i][0];
                }
            }
        }

        weak_classifiers[k].parity = parity[k];
        weak_classifiers[k].thresh = thresh[k];

    }
    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/



    // free all data
    printf("\tfreeing data...\n");
    free(thresh);
    free(parity);
    free(min_err);
    free(neg);
    free(pos);

    printf("\tfreeing res...\n");
    for (int i=0; i<total_features; ++i) {
        for(int j=0; j<weight_len; ++j) {
            free(res[i][j]);
        }
        free(res[i]);
    }

    /*
    printf("\tfreeing images_h...\n");
    for (int i=0; i<total_features; ++i) {
        free(images_h[i]);
    }
    */

    printf("\tfreeing cm_pos...\n");
    for (int i=0; i<total_features; ++i) {
        free(cm_pos[i]);
    }

    printf("\tfreeing cm_neg...\n");
    for (int i=0; i<total_features; ++i) {
        free(cm_neg[i]);
    }

    free(cm_pos);
    free(cm_neg);
    //free(images_h);
    free(res);


    printf("\tClassifying samples...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/

    block_size = 1024;
    grid_size = total_features;

    // set kernel variables
    lab_bytes = sizeof(int) * data.total_samples;
    wt_bytes = sizeof(float) * data.total_samples;
    wc_bytes = sizeof(struct weak_classifier) * total_features;
    err_bytes = sizeof(float) * total_features*data.total_samples;
    res_bytes = sizeof(int) * total_features*data.total_samples;
    /*
    hpitch = rows*cols*sizeof(float);
    width = (rows*cols) * sizeof(float);
    height = data.total_samples;
    */

    // allocate host data
    results_h = (int*) malloc(sizeof(int) * total_features*data.total_samples);
    errors_h = (float*) malloc(sizeof(float) * total_features*data.total_samples);

    // allocate device data
    //cudaMallocPitch(&images_d, &dpitch, width, height);
    cudaMalloc(&wcs_d, wc_bytes);
    cudaMalloc(&labels_d, lab_bytes);
    cudaMalloc(&results_d, res_bytes);

    // copy over device data

    //cudaMemcpy2D(images_d, dpitch, images_h, hpitch, width, height, cudaMemcpyHostToDevice);
    cudaMemcpy(labels_d,data.labels,lab_bytes,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(weights_d, weights, wt_bytes, 0, cudaMemcpyHostToDevice);

    // allocate space for section arrays of features
    wcs_h = (struct weak_classifier*) malloc(sizeof(struct weak_classifier) * total_features);

    for (int i=0; i<total_features; ++i) {
        num_sects = weak_classifiers[i].feat.num_sections;
        sect_bytes = sizeof(struct section) * num_sects;
        struct section *sects;

        cudaMalloc(&sects,sect_bytes);
        cudaMemcpy(sects,weak_classifiers[i].feat.sections, sect_bytes, cudaMemcpyHostToDevice);
        wcs_h[i].parity = weak_classifiers[i].parity;
        wcs_h[i].thresh = weak_classifiers[i].thresh;
        wcs_h[i].feat.num_sections = weak_classifiers[i].feat.num_sections;
        wcs_h[i].feat.sections = sects;
    }
    cudaMemcpy(wcs_d, wcs_h, wc_bytes, cudaMemcpyHostToDevice);

    err_bytes = sizeof(float) * total_features*data.total_samples;
    cudaMalloc(&errors_d, err_bytes);

    // launch kernel
    total_iters = (int) ceil(data.total_samples/(float)block_size);
    for (int i=0; i<total_iters; ++i) {
        classify_shared<<<grid_size,block_size>>>(errors_d, results_d, wcs_d, 
                images_d, labels_d, rows, dpitch, data.total_samples, total_features, i);
    }

    // grab error if kernel does not launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,"An error occured during launch of the kernel: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // copy errors and results to host
    cudaMemcpy(errors_h, errors_d, err_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(results_h, results_d, res_bytes, cudaMemcpyDeviceToHost);


    for (int i=0; i<total_features; ++i) {
        errors[i] = 0.0;
        for (int j=0; j<data.total_samples; ++j) {
            results[i][j] = results_h[(i*data.total_samples) + j];
            errors[i] += errors_h[(i*data.total_samples) + j];
        }
    }

    // free all device data
    for (int i=0; i<total_features; ++i) {
        cudaFree(wcs_h[i].feat.sections);
    }
    cudaFree(wcs_d);
    cudaFree(errors_d);
    cudaFree(images_d);
    cudaFree(labels_d);
    free(wcs_h);

    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/

    return errors;
}

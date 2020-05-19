#ifndef _TRAINING_H_
#define _TRAINING_H_

#include "adaboost.h"

// kernels

__global__ void train(float *res, struct weak_classifier *weak_classifiers, float *images, int rows, size_t pitch, int total_samples, int total_features);
__global__ void train_shared(float *res, struct weak_classifier *weak_classifiers, float *images, int rows, size_t pitch, int total_samples, int total_features, int iteration);
__global__ void classify(float *errors, int *results, struct weak_classifier *weak_classifiers, float *images, int *labels, int rows, size_t pitch, int total_samples, int total_features);
__global__ void classify_shared(float *errors, int *results, struct weak_classifier *weak_classifiers, float *images, int *labels, int rows, size_t pitch, int total_samples, int total_features, int iteration);


// main functions
float* train_weak_d(int **results, int total_features, struct weak_classifier *weak_classifiers, float *weights, struct data_t data, float neg_sum, float pos_sum);
float* train_weak_d_shared(int **results, int total_features, struct weak_classifier *weak_classifiers, float *weights, struct data_t data, float neg_sum, float pos_sum);

#endif

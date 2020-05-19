#ifndef _ADABOOST_H_
#define _ADABOOST_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <dirent.h>
#include "imagelib_cpu.h"

#define MAX_FEATURES 180000
#define MAX_SAMPLES 20000

struct Mat {
    float *values;
    int rows;
    int cols;
};

struct section {
    int row;
    int col;
    int height;
    int width;
    int sign;
};

struct feature {
    struct section *sections;
    int num_sections;
};

struct data_t {
    struct Mat *images;
    int *labels;
    int total_samples;
    int positive_samples;
    int negative_samples;
};

struct weak_classifier {
    struct feature feat;
    int parity;
    int thresh;
};

struct strong_classifier {
    struct weak_classifier *weak_classifiers;
    int size;
};


// function declarations
struct data_t load_data();
struct Mat read_image(char *img_path);
struct Mat* compute_integrals(struct Mat *img, int total_samples);
int shuffle(struct data_t *data);
int sort_pair(const void *a, const void *b);

int build_features(struct feature *features, int rows, int cols);

int classify_weak(struct weak_classifier wc, struct Mat img);
float apply_feat(struct feature feat, struct Mat img);
struct strong_classifier* adaboost(struct feature *features, struct data_t data, int total_features, int total_classifiers);
float* train_weak(int **results, int total_features, struct weak_classifier *weak_classifiers, float *weights, struct data_t data, float neg_sum, float pos_sum);

// visualization functions
void print_weak_classifiers(struct weak_classifier *weak_classifiers, int total_classifiers, int num_classifiers);
void print_image(struct Mat img);
void print_feature(struct feature feat,int rows,int cols);

// data cleanup
void free_mats(struct Mat *images, int total_samples);
void free_sections(struct section *sections);
void free_features(struct feature *features, int total_features);
void free_data(struct data_t data);
void free_weak_classifiers(struct weak_classifier *weak_classifiers, int total_classifiers);
void free_strong_classifiers(struct strong_classifier *strong_classifiers, int total_classifiers);

#endif

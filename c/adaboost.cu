#include "adaboost.h"

// custom sort function
int sort_pair(const void *a, const void *b) { 
    float *res_a = *(float**)a;
    float *res_b = *(float**)b;

    return res_a[0] > res_b[0];
}
struct data_t load_data(char *data_path) {
    struct data_t data;
    struct Mat *images = (struct Mat*) malloc(sizeof(struct Mat) * MAX_SAMPLES);
    data.labels = (int*) malloc(sizeof(int) * MAX_SAMPLES);

    data.total_samples = 0;
    data.negative_samples = 0;
    data.positive_samples = 0;

    DIR *dir;
    struct dirent *ent;

    char face_path[256];
    char noface_path[256];
    char img_path[512];
    int index = 0;

    strcpy(face_path,data_path);
    strcpy(noface_path,data_path);

    if((dir = opendir(data_path)) == NULL) {
        fprintf(stderr,"Error: Cannot open directory %s.\n",data_path);
        exit(EXIT_FAILURE);
    }

    while((ent = readdir(dir)) != NULL) {
        if ((ent->d_name[0] != '.') && (ent->d_type == DT_DIR)) {
            if (strcmp(ent->d_name,"face") == 0) {
                strcat(face_path,"face/");
            } else if (strcmp(ent->d_name,"no_face") == 0) {
                strcat(noface_path,"no_face/");
            }
        }
    }

    closedir(dir);

    // read in face data
    if((dir = opendir(face_path)) == NULL) {
        fprintf(stderr,"Error: Cannot open directory %s.\n",face_path);
        exit(EXIT_FAILURE);
    }

    while((ent = readdir(dir)) != NULL) {
        if ((ent->d_name[0] != '.') && (ent->d_type != DT_DIR)) {
            printf("loading positive sample %d\n", index+1);
            strcpy(img_path,face_path);
            strcat(img_path,ent->d_name);
            
            images[index] = read_image(img_path);

            data.total_samples++;
            data.positive_samples++;
            data.labels[index] = 1;

            index++;
        }
        /*
        if(index >= 100) {
            break;
        }
        */
    }

    closedir(dir);


    // read in non-face data
    if((dir = opendir(noface_path)) == NULL) {
        fprintf(stderr,"Error: Cannot open directory %s.\n",face_path);
        exit(EXIT_FAILURE);
    }

    while((ent = readdir(dir)) != NULL) {
        if ((ent->d_name[0] != '.') && (ent->d_type != DT_DIR)) {
            printf("loading negative sample %d\n", index+1);
            strcpy(img_path,noface_path);
            strcat(img_path,ent->d_name);

            images[index] = read_image(img_path);

            data.total_samples++;
            data.negative_samples++;
            data.labels[index] = 0;

            index++;
        }
        if(index >= 8000) {
            break;
        }
        /*
        if(data.negative_samples >= data.positive_samples) {
            break;
        }
        */
    }

    closedir(dir);

    printf("Computing integrals (CPU)...\n");

    /********** start time **********/
    double total_msecs = 0.0;
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    /********** start time **********/

    //compute_integrals_d(images, data.total_samples);
    data.images = compute_integrals(images,data.total_samples);
    
    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/
    //data.images = images;
    
    printf("Shuffling data...\n");

    /********** start time **********/
    gettimeofday(&start, NULL);
    /********** start time **********/

    shuffle(&data);

    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/

    // free initial images (for CPU code)
    free_mats(images,data.total_samples);

    return data;
}

struct Mat read_image(char *img_path) {
    int rows;
    int cols;
    int maxval;

    struct Mat img;

    img.values = read_ppm(img_path, &rows, &cols, &maxval);
    img.rows = rows;
    img.cols = cols;

    return img;
}

struct Mat* compute_integrals(struct Mat *images, int total_samples) {
    int rows;
    int cols;
    struct Mat *integral_images;

    rows = images[0].rows;
    cols = images[0].cols;

    float row_sum[rows][cols];
    
    integral_images = (struct Mat*) malloc(sizeof(struct Mat) * total_samples);

    for (int i=0; i<total_samples; ++i) {
        integral_images[i].values = (float*) malloc(sizeof(float) * rows*cols);
        integral_images[i].rows = images[i].rows;
        integral_images[i].cols = images[i].cols;

        for (int r=0; r<rows; ++r){
            for (int c=0; c<cols; ++c) {
                row_sum[r][c] = images[i].values[(r*rows)+c];

                if (c>0) {
                    row_sum[r][c] += row_sum[r][c-1];
                }

                integral_images[i].values[(r*rows)+c] = row_sum[r][c];

                if (r>0) {
                    integral_images[i].values[(r*rows)+c] += integral_images[i].values[((r-1)*rows)+c];
                }
            }
        }
    }
    
    return integral_images;
}

int shuffle(struct data_t *data) {
    int total_samples = data->total_samples;
    int swap_ind = 0;
    int test;
    struct Mat temp_img;
    int temp_lab;

    for (int i=0; i<total_samples; ++i) {
        swap_ind =(int) (rand() % total_samples);

        temp_img = data->images[i];
        temp_lab = data->labels[i];

        data->images[i] = data->images[swap_ind];
        data->labels[i] = data->labels[swap_ind];

        data->images[swap_ind] = temp_img;
        data->labels[swap_ind] = temp_lab;

        if(i==1020) {test=swap_ind;}
    }
    return test;
}

int build_features(struct feature *features, int rows, int cols) {
    int index = 0;
    for (int r=1; r<rows; ++r) {
        for (int c=1; c<cols; ++c) {

            // 2-section horizontal features
            for (int i=1; i+r < rows; ++i) {
                for (int j=2; j+c < cols; j+=2) {
                    int sec_w = j/2;
                    int h = i;

                    // horizontal 2-section left- right+
                    struct feature h_feat;
                    h_feat.sections = (struct section*) malloc(sizeof(struct section) * 2);
                    h_feat.num_sections = 2;

                    struct section left;
                    struct section right;

                    // create sections of feature window
                    left.row = r;
                    left.col = c;
                    left.width = sec_w;
                    left.height = h;
                    left.sign = -1;

                    
                    right.row = r;
                    right.col = c+sec_w;
                    right.width = sec_w;
                    right.height = h;
                    right.sign = 1;

                    // add sections to feature
                    h_feat.sections[0] = left;
                    h_feat.sections[1] = right;
                    

                    // add feature to features
                    features[index++] = h_feat;
                }
            }

            // 2-section vertical features
            for (int i=2; i+r < rows; i+=2) {
                for (int j=1; j+c < cols; ++j) {
                    int w = j;
                    int sec_h = i/2;

                    // vertical 2-section top- bottom+
                    struct feature v_feat;
                    v_feat.sections = (struct section*) malloc(sizeof(struct section) * 2);
                    v_feat.num_sections = 2;

                    struct section top;
                    struct section bot;

                    // create sections of feature window
                    top.row = r;
                    top.col = c;
                    top.width = w;
                    top.height = sec_h;
                    top.sign = -1;

                    bot.row = r+sec_h;
                    bot.col = c;
                    bot.width = w;
                    bot.height = sec_h;
                    bot.sign = 1;

                    // add sections to feature
                    v_feat.sections[0] = top;
                    v_feat.sections[1] = bot;

                    // add feature to features
                    features[index++] = v_feat;
                }
            }

            // 3-section horizontal features
            for (int i=1; i+r < rows; ++i) {
                for (int j=3; j+c < cols; j+=3) {
                    int sec_w = j/3;
                    int h = i;

                    // horizontal 3-section outer- inner+
                    struct feature h_feat;
                    h_feat.sections = (struct section*) malloc(sizeof(struct section) * 3);
                    h_feat.num_sections = 3;

                    struct section left;
                    struct section mid;
                    struct section right;

                    // create sections of feature window
                    left.row = r;
                    left.col = c;
                    left.height = h;
                    left.width = sec_w;
                    left.sign = -1;

                    mid.row = r;
                    mid.col = c+sec_w;
                    mid.height = h;
                    mid.width = sec_w;
                    mid.sign = 1;

                    right.row = r;
                    right.col = c + (2*sec_w);
                    right.height = h;
                    right.width = sec_w;
                    right.sign = -1;

                    // add sections to feature
                    h_feat.sections[0] = left;
                    h_feat.sections[1] = mid;
                    h_feat.sections[2] = right;

                    // add feature to features
                    features[index++] = h_feat;
                }
            }

            // 3-section vertical features
            for (int i=3; i+r < rows; i+=3) {
                for (int j=1; j+c < cols; ++j) {
                    int w = j;
                    int sec_h = i/3;

                    // vertical 3-section outer- inner+
                    struct feature v_feat;
                    v_feat.sections = (struct section*) malloc(sizeof(struct section) * 3);
                    v_feat.num_sections = 3;

                    struct section top;
                    struct section mid;
                    struct section bot;

                    // create sections of feature window
                    top.row = r;
                    top.col = c;
                    top.width = w;
                    top.height = sec_h;
                    top.sign = -1;

                    mid.row = r+sec_h;
                    mid.col = c;
                    mid.width = w;
                    mid.height = sec_h;
                    mid.sign = 1;

                    bot.row = r+(2*sec_h);
                    bot.col = c;
                    bot.width = w;
                    bot.height = sec_h;
                    bot.sign = -1;

                    // add sections to feature
                    v_feat.sections[0] = top;
                    v_feat.sections[1] = mid;
                    v_feat.sections[2] = bot;

                    // add feature to features
                    features[index++] = v_feat;
                }
            }

            // 4-section features
            for (int i=2; i+r < rows; i+=2) {
                for (int j=2; j+c < cols; j+=2) {
                    int sec_w = j/2;
                    int sec_h = i/2;

                    // diagonal 4-section top_left- top_right+
                    struct feature d_feat;
                    d_feat.sections = (struct section*) malloc(sizeof(struct section) * 4);
                    d_feat.num_sections = 4;

                    struct section tl;
                    struct section tr;
                    struct section bl;
                    struct section br;

                    // create sections of feature window
                    tl.row = r;
                    tl.col = c;
                    tl.height = sec_h;
                    tl.width = sec_w;
                    tl.sign = -1;

                    tr.row = r;
                    tr.col = c+sec_w;
                    tr.height = sec_h;
                    tr.width = sec_w;
                    tr.sign = 1;

                    bl.row = r+sec_h;
                    bl.col = c;
                    bl.height = sec_h;
                    bl.width = sec_w;
                    bl.sign = 1;

                    br.row = r+sec_h;
                    br.col = c+sec_w;
                    br.height = sec_h;
                    br.width = sec_w;
                    br.sign = -1;

                    // add sections to feature
                    d_feat.sections[0] = tl;
                    d_feat.sections[1] = tr;
                    d_feat.sections[2] = bl;
                    d_feat.sections[3] = br;

                    // add feature to features
                    features[index++] = d_feat;
                }
            }
        }
    }

    return index;
}



int classify_weak(struct weak_classifier wc, struct Mat img) {
    float res = apply_feat(wc.feat,img);
    int label = -1;
    
    if ((res * wc.parity) < (wc.thresh * wc.parity)) {
        label = 1;
    } else {
        label = 0;
    }
    return label;
}

float apply_feat(struct feature feat, struct Mat img) {
    float sum = 0;
    int A=0;
    int B=0;
    int C=0;
    int D=0;
    int rows = img.rows;
    int num_sections = feat.num_sections;
    struct section sect;

    for (int i=0; i<num_sections; ++i) {
        sect = feat.sections[i];
        A = ((sect.row*rows) + sect.col) - (rows + 1);
        B = ((sect.row*rows) + (sect.col+sect.width-1)) - rows;
        C = (((sect.row+sect.height-1)*rows) + sect.col) - 1;
        D = (((sect.row+sect.height-1)*rows) + (sect.col+sect.width-1));

        if (sect.row==0 && sect.col==0) {
            sum += sect.sign * img.values[D];
        } else if(sect.col == 0) {
            sum += sect.sign * (img.values[D] - img.values[B]);
        } else if (sect.row == 0) {
            sum += sect.sign * (img.values[D] - img.values[C]);
        } else {
            sum += sect.sign * (img.values[D] - img.values[B] - img.values[C] + img.values[A]);
        }
        
    }

    return sum;
}

struct strong_classifier* adaboost(struct feature *features, struct data_t data, int total_features, int total_classifiers) {
    float weight_sum = 0.0;
    float min_error = 0.0;
    int best_class = 0;
    float neg_sum = 0.0;
    float pos_sum = 0.0;

    struct strong_classifier *strong_classifiers;
    struct weak_classifier best_classifiers[total_classifiers];
    
    float *errors;
    int **results = (int**)malloc(sizeof(int*) * total_features);
    float *weights = (float*) malloc(sizeof(float) * data.total_samples);
    struct weak_classifier *weak_classifiers = (struct weak_classifier*) malloc(sizeof(struct weak_classifier) * total_features);

    if (data.negative_samples < 1 || data.positive_samples < 1) {
        fprintf(stderr,"Error: need to have both negative and positive samples in the dataset.\n");
        exit(EXIT_FAILURE);
    }

    // initialize weight vector
    for (int i=0; i<data.total_samples; ++i) {
        // negative sample
        if (data.labels[i] == 0) {
            weights[i] = (1.0 / (2.0*data.negative_samples));
        }
        // positive sample
        else {
            weights[i] = (1.0 / (2.0*data.positive_samples));
        }
    }

    // build vector of initial weak classifiers
    for (int k=0; k<total_features; ++k) {
        // allocate for each feature results
        results[k] = (int*) malloc(sizeof(int) * data.total_samples);

        struct weak_classifier wc;
        wc.feat = features[k];
        wc.parity = 0;
        wc.thresh = 0;
        weak_classifiers[k] = wc;
    }

    // select 'total_classifiers' weak classifiers
    for (int t=0; t<total_classifiers; ++t) {
        printf("Getting classifier #%d\n",t);

        // normalize the weights & sum class weights
        weight_sum = 0.0;
        neg_sum = 0.0;
        pos_sum = 0.0;

        for (int i=0; i<data.total_samples; ++i) {
            weight_sum += weights[i];
        }
        for (int i=0; i<data.total_samples; ++i) {
            weights[i] = (weights[i]/weight_sum);

            // sum negative and positive weights
            if (data.labels[i] == 0) {
                neg_sum += weights[i];
            } else {
                pos_sum += weights[i];
            }
        }

        // iterate through all features
        /********** start time **********/
        double total_msecs = 0.0;
        struct timeval stop, start;
        gettimeofday(&start, NULL);
        /********** start time **********/

        errors = train_weak_d_shared(results, total_features, weak_classifiers, weights, data, neg_sum, pos_sum);
        
        /********** end time **********/
        gettimeofday(&stop, NULL);
        total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
        printf("Total training time:\n\t%0.8f ms.\n", total_msecs/1000.0);
        /********** end time **********/


        min_error = errors[0];
        best_class = 0;

        for (int e=1; e<total_features; ++e) {
            if (errors[e] < min_error) {
                min_error = errors[e];
                best_class = e;
            }
        }

        printf("Best classifier: %d\nError: %f\n\n",best_class,min_error);
        
        // store best classifier
        best_classifiers[t] = weak_classifiers[best_class];

        // remove best weak classifier
        for (int i=best_class+1,j=best_class; i<total_features; ++i,++j) {
            weak_classifiers[j] = weak_classifiers[i];
        }

        // decrement number of features to choose from
        total_features--;

        
        // update weights
        for (int i=0; i<data.total_samples; ++i) {
            weights[i] = weights[i] * pow((min_error/(1-min_error)), (1-results[best_class][i]));
        }
    }

    // free arrays
    for (int i=0; i<total_features; ++i) {
        free(results[i]);
    }

    free(results);
    free(weights);
    free(weak_classifiers);
    free(errors);

    // print out best n feature visuals
    /*
    int num_classifiers = 4;
    print_weak_classifiers(best_classifiers,total_classifiers,num_classifiers);
    */

    // create strong classifiers
    
    return strong_classifiers;
}


float* train_weak(int **results, int total_features, struct weak_classifier *weak_classifiers, float *weights, struct data_t data, float neg_sum, float pos_sum) {

    int A=0, B=0, C=0, D=0;
    int label = -1;
    int num_sects;
    int rows = data.images[0].rows;
    int cols = data.images[0].cols;
    int weight_len = data.total_samples;
    float sum = 0.0, error = 0.0, cm_neg_sum = 0.0, cm_pos_sum = 0.0;
    double total_msecs = 0.0;
    struct timeval stop, start;

    int *thresh = (int*) malloc(sizeof(int) * total_features);
    int *parity = (int*) malloc(sizeof(int) * total_features);
    float *min_err = (float*) malloc(sizeof(float)*total_features);
    float *neg = (float*) malloc(sizeof(float) * total_features);
    float *pos = (float*) malloc(sizeof(float) * total_features);
    float *errors = (float*) malloc(sizeof(float) * total_features);

    float **cm_pos = (float**) malloc(sizeof(float*) * total_features);
    float **cm_neg = (float**) malloc(sizeof(float*) * total_features);

    float ***res = (float***) malloc(sizeof(float**) * total_features);

    for (int i=0; i<total_features; ++i) {
        res[i] = (float**) malloc(sizeof(float*) * weight_len);
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

    for (int k=0; k<total_features; ++k) {
        for (int i=0; i<data.total_samples; ++i) {
            struct feature feat = weak_classifiers[k].feat;
            struct Mat img = data.images[i];

            int num_sections = feat.num_sections;
            struct section sect;
            sum = 0.0;

            for (int s=0; s<num_sections; ++s) {
                sect = feat.sections[s];
                A = ((sect.row*rows) + sect.col) - (rows + 1);
                B = ((sect.row*rows) + (sect.col+sect.width-1)) - rows;
                C = (((sect.row+sect.height-1)*rows) + sect.col) - 1;
                D = (((sect.row+sect.height-1)*rows) + (sect.col+sect.width-1));

                sum += sect.sign * (img.values[D] - img.values[B] - img.values[C] + img.values[A]);
            }

            res[k][i][0] = sum;
            res[k][i][1] = weights[i];
        }
    }
    /********** end time *********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/
    /********** CPU (train) **********/

    
    /********** CPU **********/
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
    /********** CPU **********/


    printf("\tBuidling cumulative weights...\n");
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

    // free all data
    free(thresh);
    free(parity);
    free(min_err);
    free(neg);
    free(pos);
    for (int i=0; i<total_features; ++i) {
        for(int j=0; j<weight_len; ++j) {
            free(res[i][j]);
        }
        free(res[i]);
        free(cm_pos[i]);
        free(cm_neg[i]);
    }
    free(cm_pos);
    free(cm_neg);
    free(res);


    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/

    printf("\tClassifying samples...\n");
    /********** start time **********/
    total_msecs = 0.0;
    gettimeofday(&start, NULL);
    /********** start time **********/
    for (int k=0; k<total_features; ++k) {
        error = 0.0;
        struct feature feat = weak_classifiers[k].feat;
        struct section sect;
        num_sects = feat.num_sections;

        for (int i=0; i<data.total_samples; ++i) {
            struct Mat img = data.images[i];
            sum = 0.0;

            for (int s=0; s<num_sects; ++s) {
                sect = feat.sections[s];
                A = ((sect.row*rows) + sect.col) - (rows + 1);
                B = ((sect.row*rows) + (sect.col+sect.width-1)) - rows;
                C = (((sect.row+sect.height-1)*rows) + sect.col) - 1;
                D = (((sect.row+sect.height-1)*rows) + (sect.col+sect.width-1));

                sum += sect.sign * (img.values[D] - img.values[B] - img.values[C] + img.values[A]);
            }

            label = (sum * weak_classifiers[k].parity) < (weak_classifiers[k].thresh * weak_classifiers[k].parity);

            results[k][i] = abs(label-(data.labels[i]));
            error += (weights[i] * results[k][i]);
        
        }

        errors[k] = error;
    }
    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/



    for (int i=0; i<total_features; ++i) {
        for(int j=0; j<weight_len; ++j) {
            free(res[i][j]);
        }
        free(res[i]);
        free(cm_pos[i]);
        free(cm_neg[i]);
    }
    free(cm_pos);
    free(cm_neg);
    free(res);

    return errors;
}

void print_image(struct Mat img) {
    int rows = img.rows;
    int cols = img.cols;

    for (int r=0; r<rows; ++r){
        printf("| ");
        for (int c=0; c<cols-1; ++c) {
            printf("%.1f ",img.values[(r*rows)+c]);
        }
        printf("%.1f |\n",img.values[(r*rows)+cols-1]);
    }                
}

void print_feature(struct feature feat,int rows,int cols) {
    for(int j=0;j<feat.num_sections;++j){
        printf("section %d: (r=%d,c=%d,h=%d,w=%d,s=%d)\n",j+1,feat.sections[j].row,feat.sections[j].col,feat.sections[j].height,feat.sections[j].width,feat.sections[j].sign);
    }
    printf("\n");
}

void print_weak_classifiers(struct weak_classifier *weak_classifiers, int total_classifiers, int num_classifiers) {
    int rows = 24;
    int cols = 21;

    /********** print n features **********/
    if (num_classifiers > total_classifiers){
        fprintf(stderr, "Warning: cannot print out more classifiers than the amount that were created:\n\tRequested visuals: %d\n\tExisting classifiers: %d\nPrinting out all %d classifiers:\n\n", num_classifiers,total_classifiers,total_classifiers);
    }

    for (int i=0; i<total_classifiers; ++i) {
        if(i > num_classifiers) {
            break;
        }

        printf("Classifier #%d\n",i+1);
        printf("parity: %d\nthreshold: %d\n",weak_classifiers[i].parity,weak_classifiers[i].thresh);
        print_feature(weak_classifiers[i].feat,rows,cols);
    }
}

void free_mats(struct Mat *images, int total_samples) {

    for (int i=0; i<total_samples; ++i) {
        free(images[i].values);
    }

    free(images);
}

void free_features(struct feature *features, int total_features) {
    printf("Cleaning up %d features...\n",total_features);

    for (int i=0; i<total_features; ++i) {
        free(features[i].sections);
    }

    free(features);
}

void free_data(struct data_t data) {
    int total_samples = data.total_samples;
    printf("Cleaning up %d samples...\n",total_samples);

    free_mats(data.images, total_samples);
    free(data.labels);
}

void free_weak_classifiers(struct weak_classifier *weak_classifiers, int total_classifiers) {
    printf("Cleaning up %d weak classifiers...\n", total_classifiers);

    for (int i=0; i<total_classifiers; ++i) {
       free(weak_classifiers[i].feat.sections);
    }

    free(weak_classifiers);
}

void free_strong_classifiers(struct strong_classifier *strong_classifiers, int total_classifiers) {
    printf("Cleaning up %d strong classifiers...\n", total_classifiers);

    for (int i=0; i<total_classifiers; ++i) {
       free_weak_classifiers(strong_classifiers[i].weak_classifiers, strong_classifiers[i].size); 
    }

    free(strong_classifiers);
}

int main(int argc, char **argv) {
    int rows;
    int cols;
    int total_features;
    struct feature *features = (struct feature*) malloc(sizeof(struct feature) * MAX_FEATURES);
    struct data_t data;
    char *data_path;
    srand(time(0));

    if (argc != 2) {
        fprintf(stderr,"Error: usage %s <data-path>\n",argv[0]);
        return EXIT_FAILURE;
    }

    data_path = argv[1];

    /***** STEP 1: load in the data *****/
    printf("Loading in data...\n");
    data = load_data(data_path);

    // grab image shape
    struct Mat sample_image = data.images[0];
    rows = sample_image.rows;
    cols = sample_image.cols;
    
    /***** STEP 2: build feature set (filter bank) *****/
    printf("Building the feature set...\n");

    /********** start time **********/
    double total_msecs = 0.0;
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    /********** start time **********/

    total_features = build_features(features,rows,cols);

    /********** end time **********/
    gettimeofday(&stop, NULL);
    total_msecs = (double) (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\t%0.8f ms.\n", total_msecs/1000.0);
    /********** end time **********/

    /***** STEP 3: run AdaBoosting through data with feature_set (filter bank) *****/

    // specify the total number of classifiers to include for all strong classifiers (6061 in V&J paper)
    int total_classifiers = 1;

    struct strong_classifier *classifiers = adaboost(features,data,total_features,total_classifiers);

    // free up memory
    free_features(features,total_features);
    free_data(data);
    //free_strong_classifiers(classifiers);

    return EXIT_SUCCESS;
}

#ifndef _IMAGELIB_H_
#define _IMAGELIB_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float *read_ppm( char *filename, int * rows, int * cols, int *maxval );
void write_ppm( char *filename, int rows, int cols, int maxval, float *pic);

#endif

#include "imagelib_cpu.h"

/*
 * Read a ppm image from the filename provided
 */
float *read_ppm(char *filename, int *rows, int *cols, int *maxval){
  
	if (!filename || filename[0] == '\0') {
		fprintf(stderr, "ERROR: No file name provided to read_ppm().\n");
		exit(EXIT_FAILURE);
	}

	FILE *fp;

	fp = fopen( filename, "rb");
	if (!fp) {
            fprintf(stderr, "ERROR: File '%s' cannot be opened for reading.\n", filename);
            exit(EXIT_FAILURE);
	}

	char chars[1024];
	int num = fread(chars, sizeof(char), 1000, fp);

	if (chars[0] != 'P' || chars[1] != '6') {
		fprintf(stderr, "ERROR: File '%s' does not start with \"P6\". Expecting a binary PPM file.\n", filename);
                exit(EXIT_FAILURE);
	}

	unsigned int width, height, maxvalue;


	char *ptr = chars+3;
	if (*ptr == '#') {
            ptr = 1 + strstr(ptr, "\n");
	}

	num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
	*cols = width;
	*rows = height;
	*maxval = maxvalue;
  
	float *pic = (float*) malloc( width * height * sizeof(float));

	if (!pic) {
            fprintf(stderr, "ERROR: Unable to allocate %d x %d unsigned ints for the picture.\n", width, height);
            exit(EXIT_FAILURE);
	}

	// allocate buffer to read the rest of the file into
	int bufsize =  3 * width * height * sizeof(unsigned char);
	if ((*maxval) > 255) {
            bufsize *= 2;
        }

	unsigned char *buf = (unsigned char *)malloc( bufsize );
	if (!buf) {
            fprintf(stderr, "ERROR: Unable to allocate %d bytes of read buffer\n", bufsize);
            exit(EXIT_FAILURE);
	}

	// really read
	char duh[80];
	char *line = chars;

	// find the start of the pixel data. 
	sprintf(duh, "%d", *cols);
	line = strstr(line, duh);
	line += strlen(duh) + 1;

	sprintf(duh, "%d", *rows);
	line = strstr(line, duh);
	line += strlen(duh) + 1;

	sprintf(duh, "%d", *maxval);
	line = strstr(line, duh);
	line += strlen(duh) + 1;

	long offset = line - chars;
	fseek(fp, offset, SEEK_SET);

	long numread = fread(buf, sizeof(char), bufsize, fp);

	fclose(fp);
	
	int pixels = (*cols) * (*rows);
	for (int i=0; i<pixels; i++) {
            pic[i] = (float) buf[3*i];  // red channel... might need to change for grayscale
        }

        free(buf);
	
	return pic;
}

/*
 * Write a ppm image to the filename provided
 */
void write_ppm( char *filename, int cols, int rows, int maxval, float *pic)
{
	FILE *fp;
  
	fp = fopen(filename, "wb");
	if (!fp) {
            fprintf(stderr, "ERROR: File '%s' cannot be opened for writting.\n", filename);
            exit(EXIT_FAILURE);
	}
  
	fprintf(fp, "P6\n"); 
	fprintf(fp,"%d %d\n%d\n", cols, rows, maxval);
  
	int numpix = cols * rows;
	for (int i=0; i<numpix; i++) {
		unsigned char uc = (unsigned char) pic[i];
		fprintf(fp, "%c%c%c", uc, uc, uc); 
	}

	fclose(fp);
}

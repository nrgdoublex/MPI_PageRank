#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <sys/time.h>

typedef struct {
  char *input_name;
  char *output_name;
  double time_limit;
  int max_iter;
  int verbose;
} Options;

/* Data structures for timing */
double start_time, end_time;
struct timeval tz;
struct timezone tx;

void init_options(Options **options);
int read_options(int argc, char **argv, Options *options);
double sup_norm(int size, double *v1, double *v2);
void get_sparse_matrix( char *file_name, int *n, int *m, double **value,
                        int **colind, int **rbegin, int verbose);

void get_matrix(char *file_name, int *n, int *m, int **rowind,
                int **colind, double **values, int verbose);

void debug_print_int(int process, int size, int *arr);
void debug_print_double(int process, int size, double *arr);

#endif

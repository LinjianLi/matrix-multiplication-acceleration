#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <locale.h>
#include "omp.h"
#include "GenMatrix.h"
#include "MatrixCalcu.h"


int main(int argc, char *argv[]) {

  FILE *log_file;
  
  int n = strtol(argv[1]);
  float seed = strtod(argv[2]);
  int num_threads = strtol(argv[3]);

  log_file = fopen("program-log-Blocked-OMP.txt", "w");
  fprintf(log_file, "Blocked, OpenMP, %d Threads\n----------\n", num_threads);
  fprintf(log_file, "Number of cores: %d\n----------\n", omp_get_num_procs());

  float *a = malloc(n*n * sizeof(*a));  // float *a = new float[n*n];
  float *b = malloc(n*n * sizeof(*b));
  float *c;

  matrix_gen(a,b,n,seed);
  fprintf(log_file, "Order of matrix: %d\nSeed: %f\n----------\n", n, seed);

  fflush(log_file);


  for (int block_size=8; block_size<=128; block_size<<=1) {

    struct timeval start;
    struct timeval end;
    double diff;
    gettimeofday(&start,NULL);

    c = SquareMatrixMul_Blocked_OMP(a,b,n,block_size,num_threads);
    float trace = SquareMatrixTrace(c,n);

    fprintf(log_file, "%dx%d Blocked\n", block_size, block_size);
    fprintf(log_file, "Trace: %f\n", trace);

    // Calculate the time spent by the multiplication.
    gettimeofday(&end,NULL);
    diff = (end.tv_sec-start.tv_sec) + ((double)(end.tv_usec-start.tv_usec))/1000000;
    setlocale(LC_NUMERIC, "");
    fprintf(log_file, "The time spent is %'lf seconds\n----------\n", diff);
    fflush(log_file);
    // diff is time spent by the program, and the unit is second

    free(c);
  }

  fprintf(log_file, "Finished\n);
  fclose(log_file);

  return 0;
}

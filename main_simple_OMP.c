#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <locale.h>
#include "omp.h"
#include "GenMatrix.h"
#include "MatrixCalcu.h"


int main(int argc, char *argv[]) {

  FILE *log_file;
  
  const int n = 8192;
  const float seed = 0.3;
  log_file = fopen("program-log-simple-OMP.txt", "w");
  fprintf(log_file, "Simple OpenMP\n----------\n");
  fprintf(log_file, "Number of cores: %d\n----------\n", omp_get_num_procs());

  float *a = malloc(n*n * sizeof(*a));  // float *a = new float[n*n];
  float *b = malloc(n*n * sizeof(*b));
  float *c;

  matrix_gen(a,b,n,seed);
  fprintf(log_file, "Order of matrix: %d\nSeed: %f\n----------\n", n, seed);

  fflush(log_file);

  int arr_num_threads[5] = {2,4,8,12,16};
  for (int i=0; i<5; ++i) {

    struct timeval start;
    struct timeval end;
    double diff;
    gettimeofday(&start,NULL);

    c = SquareMatrixMul_OMP(a,b,n,arr_num_threads[i]);
    float trace = SquareMatrixTrace(c,n);

    fprintf(log_file, "%d Threads\n", arr_num_threads[i]);
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

  fclose(log_file);

  return 0;
}

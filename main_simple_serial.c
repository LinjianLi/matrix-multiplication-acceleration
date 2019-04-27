#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <locale.h>
#include "GenMatrix.h"
#include "MatrixCalcu.h"


int main(int argc, char *argv[]) {

  FILE *log_file;
  
  float seed = 0.3;
  log_file = fopen("program-log-simple-serial.txt", "w");
  fprintf(log_file, "Simple Serial\n----------\n");

  
  for (int n=512; n<=8192; n<<=1) {
    float *a = malloc(n*n * sizeof(*a));  // float *a = new float[n*n];
    float *b = malloc(n*n * sizeof(*b));

    matrix_gen(a,b,n,seed);
    fprintf(log_file, "Order of matrix: %d\tSeed: %f\n", n, seed);

    struct timeval start;
    struct timeval end;
    double diff;
    gettimeofday(&start,NULL);

    float *c = SquareMatrixMul_Serial(a, b, n);
    float trace = SquareMatrixTrace(c,n);

    fprintf(log_file, "Trace: %f\n", trace);

    // Calculate the time spent by the multiplication.
    gettimeofday(&end,NULL);
    diff = (end.tv_sec-start.tv_sec) + ((double)(end.tv_usec-start.tv_usec))/1000000;
    setlocale(LC_NUMERIC, "");
    fprintf(log_file, "The time spent is %'lf seconds\n----------\n", diff);
    fflush(log_file);
    // diff is time spent by the program, and the unit is second

    free(a);
    free(b);
    free(c);
  }

  fclose(log_file);

  return 0;
}

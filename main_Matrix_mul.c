//
// Created by LinjianLi on 2019/4/28.
//

#include <stdio.h>
#include <stdlib.h>
#include "GenMatrix.h"
#include "MatrixCalcu.h"

int main(int argc, char *argv[]) {

  int n = strtol(argv[1], NULL, 10);
  float seed = strtof(argv[2], NULL);

  int num_threads = 24;

  float *a = malloc(n*n * sizeof(*a));  // float *a = new float[n*n];
  float *b = malloc(n*n * sizeof(*b));
  float *c;

  matrix_gen(a,b,n,seed);
  c = SquareMatrixMul_8x8Blocked_AVX_OMP(a,b,n,num_threads);
  float trace = SquareMatrixTrace(c,n);
  printf("%lf", trace);

  return 0;
}

//
// Created by LinjianLi on 2019/4/19.
//

#ifndef MATRIX_MULTIPLICATION_ACCELERATION_GENMATRIX_H
#define MATRIX_MULTIPLICATION_ACCELERATION_GENMATRIX_H

/**
 * Input: a, b are the N*N float matrix, 0<seed<1, float
 * This function should initialize two matrixs with rand_float()
 */

float rand_float(float s) {
  return 4*s*(1-s);
}

void matrix_gen(float *a, float *b, int N, float seed){
  for (int i=0;i<N*N;i++) {
    seed=rand_float(seed);
    a[i]=seed;
    seed=rand_float(seed);
    b[i]=seed;
  }
}

#endif //MATRIX_MULTIPLICATION_ACCELERATION_GENMATRIX_H

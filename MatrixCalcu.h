//
// Created by LinjianLi on 2019/4/17.
//

#ifndef MATRIX_MULTIPLICATION_ACCELERATION_MATRIXCALCU_H
#define MATRIX_MULTIPLICATION_ACCELERATION_MATRIXCALCU_H

#include <xmmintrin.h>  // SSE _m128
#include <immintrin.h>  // AVX _m256
#include <memory.h>
#include "omp.h"

// I want to use inline function.
// If I do not add this declaration, there will be
// "relocation truncated to fit: R_X86_64_PC32 against undefined symbol `Convert2DIndexTo1DIndex'"
// And I do not know why!!!
size_t Convert2DIndexTo1DIndex (size_t size_row, size_t index_row, size_t index_column);


inline size_t Convert2DIndexTo1DIndex (size_t size_row, size_t index_row, size_t index_column) {
  return (index_row * size_row + index_column);
}

float* InitZeroSquareMatrix(size_t block_size) {
  float *result = malloc(block_size*block_size * sizeof(*result));
  memset(result, 0, block_size*block_size*sizeof(*result));
  return result;
}

float SquareMatrixTrace(float *mtrx, int n) {
  float trace = 0;
  #pragma omp parallel for reduction(+:trace)
  for (size_t i=0; i<n*n; i+=(n+1)) {
    trace += mtrx[i];
  }
  return trace;
}

void PrintSquareMatrix(float *mtrx, size_t n) {
  #pragma omp parallel for
  for (size_t i=0; i<n; ++i) {
    for (size_t j=0; j<n; ++j) {
      printf("%.4f  ", mtrx[Convert2DIndexTo1DIndex(n,i,j)]);
    }
    printf("\n");
  }
}

void SquareMatrixAddTo(float *dest, float *src, int n) {
  #pragma omp parallel for
  for (size_t i=0; i<n*n; ++i) {dest[i] += src[i];}
}

float* SquareMatrixMul_SimpleSerial(float *a, float *b, int n) {
  float *result = malloc(n*n * sizeof(*result));  // float *result = new float[n*n];
  for (size_t i=0; i<n; ++i) {
    for (size_t j=0; j<n; ++j) {
      size_t index = Convert2DIndexTo1DIndex(n,i,j);
      result[index] = 0;
      for (size_t k=0; k<n; ++k) {
        size_t index_for_a =Convert2DIndexTo1DIndex(n,i,k);
        size_t index_for_b =Convert2DIndexTo1DIndex(n,k,i);
        result[index] += a[index_for_a]*b[index_for_b];
      }
    }
  }
  return result;
}

/**
 * @param a
 * @param b
 * @param n : must be multiple of 4
 * @return
 */
float* SquareMatrixMul_SSE(float *a, float *b, int n) {
  float *result = malloc(n*n * sizeof(*result));

  size_t temp = n*n * sizeof(float) / sizeof(__m128);

  __m128 *rows_right_matrix = malloc(temp);

  for (size_t col=0; col<n; col+=4) {
    for (size_t row=0; row<n; ++row) {
      rows_right_matrix[row] = _mm_load_ps(b+Convert2DIndexTo1DIndex(n,row,col));
    }

    for (size_t row=0; row<n; ++row) {
      __m128 sum_elem_mul = _mm_setzero_ps();
      for (size_t column=0; column<n; ++column) {
        __m128 element_left_matrix = _mm_load_ps1(a+Convert2DIndexTo1DIndex(n,row,column));
        element_left_matrix = _mm_mul_ps(element_left_matrix, rows_right_matrix[column]);
        sum_elem_mul = _mm_add_ps(sum_elem_mul, element_left_matrix);
      }
      _mm_store_ps(result+Convert2DIndexTo1DIndex(n,row,col), sum_elem_mul);
    }
  }
  return result;
}

/**
 * @param a
 * @param b
 * @param n : must be multiple of 8
 * @return
 */
float* SquareMatrixMul_AVX(float *a, float *b, int n) {

  fprintf(stderr,"This will cause segmentation fault!\nThe output is wrong!");
  return a;

  float *result = malloc(n*n * sizeof(*result));

  __m256 *rows_right_matrix = malloc(n*n * sizeof(float) / sizeof(__m256));

  for (size_t col=0; col<n; col+=8) {
    for (size_t row=0; row<n; ++row) {


      // todo: I do not know why I can not assign a __m256 variable to the elements of __m256 array.
      //       It will cause segmentation fault.
      //rows_right_matrix[row] = _mm256_load_ps(b+Convert2DIndexTo1DIndex(n,row,col));
      __m256 test = _mm256_load_ps(b+Convert2DIndexTo1DIndex(n,row,col));
      rows_right_matrix[row] = _mm256_load_ps(b+Convert2DIndexTo1DIndex(n,row,col));;


    }

    for (size_t row=0; row<n; ++row) {
      __m256 sum_elem_mul = _mm256_set1_ps(0);
      for (size_t column=0; column<n; ++column) {
        __m256 element_left_matrix = _mm256_set1_ps(a[Convert2DIndexTo1DIndex(n,row,column)]);
        element_left_matrix = _mm256_mul_ps(element_left_matrix, rows_right_matrix[column]);
        sum_elem_mul = _mm256_add_ps(sum_elem_mul, element_left_matrix);
      }
      _mm256_store_ps(result+Convert2DIndexTo1DIndex(n,row,col), sum_elem_mul);
    }
  }
  return result;
}


float* SquareMatrixMul_MultiThreadsByOMP(float *a, float *b, int n) {

  float *result = malloc(n*n * sizeof(*result));  // float *result = new float[n*n];

  #pragma omp parallel for collapse(2)
  for (size_t i=0; i<n; ++i) {
    for (size_t j=0; j<n; ++j) {
      size_t index = Convert2DIndexTo1DIndex(n,i,j);
      result[index] = 0;
      for (size_t k=0; k<n; ++k) {
        size_t index_for_a =Convert2DIndexTo1DIndex(n,i,k);
        size_t index_for_b =Convert2DIndexTo1DIndex(n,k,i);
        result[index] += a[index_for_a]*b[index_for_b];
      }
    }
  }
  return result;
}


float* SquareMatrixSelectSquareBlock(float* original_mtrx, size_t original_mtrx_size, size_t top, size_t left, size_t block_size) {
  float *result = malloc(block_size*block_size * sizeof(*result));
  #pragma omp parallel for collapse(2)
  for (size_t i=0; i<block_size; ++i) {
    for (size_t j=0; j<block_size; ++j) {
      result[Convert2DIndexTo1DIndex(block_size,i,j)]
        = original_mtrx[Convert2DIndexTo1DIndex(original_mtrx_size, i+top, j+left)];
    }
  }
  return result;
}


void FillSquareBlockInToBigSquareMatrix(float *block, size_t block_size,
                                        float *big_mtrx, size_t big_mtrx_size,
                                        size_t top, size_t left) {
  #pragma omp parallel for collapse(2)
  for (size_t i=0; i<block_size; ++i) {
    for (size_t j=0; j<block_size; ++j) {
      big_mtrx[Convert2DIndexTo1DIndex(big_mtrx_size, i+top, j+left)] =
              block[Convert2DIndexTo1DIndex(block_size,i,j)];
    }
  }
}



float* SquareMatrixMul_SplitToBlocks(float *a, float *b, int matrix_order, int block_size) {

  float *result = malloc(matrix_order*matrix_order * sizeof(*result));  // float *result = new float[matrix_order*matrix_order];

  for (size_t i=0; i<matrix_order; i+=block_size) {
    for (size_t j=0; j<matrix_order; j+=block_size) {
      float *block_result = InitZeroSquareMatrix(block_size);
      for (size_t k=0; k<matrix_order; k+=block_size) {
        float *block_a = SquareMatrixSelectSquareBlock(a,matrix_order,i,k,block_size);
        float *block_b = SquareMatrixSelectSquareBlock(b,matrix_order,k,j,block_size);
        SquareMatrixAddTo(block_result, SquareMatrixMul_SimpleSerial(block_a,block_b,block_size), block_size);
      }
      FillSquareBlockInToBigSquareMatrix(block_result,block_size,result,matrix_order,i,j);
    }
  }

  return result;
}


float* SquareMatrixMul_SplitToBlocks_MultiThreadsByOMP(float *a, float *b, int matrix_order, int block_size) {

  float *result = malloc(matrix_order*matrix_order * sizeof(*result));  // float *result = new float[matrix_order*matrix_order];
  #pragma omp parallel for collapse(2)
  for (size_t i=0; i<matrix_order; i+=block_size) {
    for (size_t j=0; j<matrix_order; j+=block_size) {
      float *block_result = InitZeroSquareMatrix(block_size);
      for (size_t k=0; k<matrix_order; k+=block_size) {
        float *block_a = SquareMatrixSelectSquareBlock(a,matrix_order,i,k,block_size);
        float *block_b = SquareMatrixSelectSquareBlock(b,matrix_order,k,j,block_size);
        SquareMatrixAddTo(block_result, SquareMatrixMul_MultiThreadsByOMP(block_a,block_b,block_size), block_size);
        //SquareMatrixAddTo(block_result, SquareMatrixMul_SimpleSerial(block_a,block_b,block_size), block_size);
      }
      FillSquareBlockInToBigSquareMatrix(block_result,block_size,result,matrix_order,i,j);

    }
  }

  return result;
}


// todo: implement
float* SquareMatrixMul_BlocksAndSSE(float *a, float *b, int matrix_order, int block_size) {

  float *result = malloc(matrix_order*matrix_order * sizeof(*result));  // float *result = new float[matrix_order*matrix_order];

  for (size_t i=0; i<matrix_order; i+=block_size) {
    for (size_t j=0; j<matrix_order; j+=block_size) {
      float *block_result = InitZeroSquareMatrix(block_size);
      for (size_t k=0; k<matrix_order; k+=block_size) {
        float *block_a = SquareMatrixSelectSquareBlock(a,matrix_order,i,k,block_size);
        float *block_b = SquareMatrixSelectSquareBlock(b,matrix_order,k,j,block_size);
        SquareMatrixAddTo(block_result, SquareMatrixMul_SSE(block_a,block_b,block_size), block_size);
      }
      FillSquareBlockInToBigSquareMatrix(block_result,block_size,result,matrix_order,i,j);
    }
  }

  return result;
}


// todo: implement
float* SquareMatrixMul_BlocksAndAVX(float *a, float *b, int matrix_order, int block_size) {

  float *result = malloc(matrix_order*matrix_order * sizeof(*result));  // float *result = new float[matrix_order*matrix_order];

  // todo: implement

  return result;
}




#endif //MATRIX_MULTIPLICATION_ACCELERATION_MATRIXCALCU_H

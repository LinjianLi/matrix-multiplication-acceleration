//
// Created by LinjianLi on 2019/4/17.
//

#ifndef MATRIX_MULTIPLICATION_ACCELERATION_MATRIXCALCU_H
#define MATRIX_MULTIPLICATION_ACCELERATION_MATRIXCALCU_H

#include <xmmintrin.h>  // SSE _m128
#include <immintrin.h>  // AVX _m256
#include <string.h>     // memset
#include <memory.h>
#include "omp.h"


float MinOfMaxOfEachRow_AVX(float *c,int N){
    float* results = malloc(sizeof(float)*8);
    for(int j = 0; j<8; j++){
        results[j]=0;
    }
    float min = 1<<20;
    for(int i =0; i<N; i++){
        float max =-1;
        __m256 result = _mm256_set1_ps(-1024.0);
        for(int j = 0; j<N; j+=8) {
            __m256 temp = _mm256_loadu_ps(&c[i*N+j]);
            result = _mm256_max_ps(result,temp);
        }
        _mm256_storeu_ps(results,result);
        for(int j = 0; j<8; j++){
            if(results[j]>max)
                max = results[j];
        }
        if(max<min){
            min=max;
        }
    }
    return min;
}

float MinOfMaxOfEachRow(float *mtrx, size_t n) {
  float *max_each_row = malloc(n * sizeof(*max_each_row));
  #pragma omp parallel for // Do not use "collapse(2)" !!!
  for (int row=0; row<n; ++row) {
    for (int column=0; column<n; ++column) {
      if (column==0 || max_each_row[row]<mtrx[row*n+column]) {
        max_each_row[row] = mtrx[row*n+column];
      }
    }
  }
  float min;
  for (int i=0; i<n; ++i) {
    if (i==0 || min>max_each_row[i]) {
      min = max_each_row[i];
    }
  }
  return min;
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
  for (size_t i=0; i<n; ++i) {
    for (size_t j=0; j<n; ++j) {
      printf("%.4f  ", mtrx[n*i+j]);
    }
    printf("\n");
  }
}

void SquareMatrixAddTo(float *dest, float *src, int n) {
  #pragma omp parallel for
  for (size_t i=0; i<n*n; ++i) {dest[i] += src[i];}
}

float* SquareMatrixMul_OMP(float *a, float *b, int n, int num_threads) {
  float *result = malloc(n*n * sizeof(*result));

  omp_set_num_threads(num_threads);
  #pragma omp parallel for collapse(2)
  for (size_t i=0; i<n; ++i) {
    for (size_t j=0; j<n; ++j) {
      int index = n*i+j;
      result[index] = 0;
      for (size_t k=0; k<n; ++k) {
        size_t index_for_a = n*i+k;
        size_t index_for_b = n*k+j;
        result[index] += a[index_for_a]*b[index_for_b];
      }
    }
  }
  return result;
}

float* SquareMatrixMul_Serial(float *a, float *b, int n) {
  return SquareMatrixMul_OMP(a, b, n, 1);
}

/**
 * @param a
 * @param b
 * @param n : must be multiple of 4
 * @return
 */
float* SquareMatrixMul_SSE(float *a, float *b, int n) {
  float *result = malloc(n*n * sizeof(*result));

  __m128 *rows_right_matrix = malloc(n * sizeof(__m128));

  for (size_t col=0; col<n; col+=4) {
    for (size_t row=0; row<n; ++row) {
      rows_right_matrix[row] = _mm_load_ps(b+ n*row+col);
    }

    for (size_t row=0; row<n; ++row) {
      __m128 sum_elem_mul = _mm_setzero_ps();
      for (size_t column=0; column<n; ++column) {
        __m128 element_left_matrix = _mm_load_ps1(a+ n*row+column);
        element_left_matrix = _mm_mul_ps(element_left_matrix, rows_right_matrix[column]);
        sum_elem_mul = _mm_add_ps(sum_elem_mul, element_left_matrix);
      }
      _mm_store_ps(result+ n*row+col, sum_elem_mul);
    }
  }
  free(rows_right_matrix);
  return result;
}


/**
 * @param a
 * @param b
 * @param matrix_order : must be multiple of 8
 * @return
 *
 * C_ij = sum(A_ik * B_kj)
 */
float* SquareMatrixMul_8x8Blocked_AVX_OMP(float *a, float *b, int matrix_order, int num_threads) {

  const int block_size = 8;

  float *result = malloc(matrix_order*matrix_order * sizeof(*result));

  omp_set_num_threads(num_threads);
  #pragma omp parallel for collapse(2)
  for (int i=0; i<matrix_order; i+=block_size) {
    for (int j=0; j<matrix_order; j+=block_size) {

      __m256 c[block_size];
      memset(c, 0, block_size*sizeof(*c));

      for (int k=0; k<matrix_order; k+=block_size) {
        __m256 a_x0,a_x1,a_x2,a_x3,a_x4,a_x5,a_x6,a_x7,
                b_0x,b_1x,b_2x,b_3x,b_4x,b_5x,b_6x,b_7x;

        int offset_b = k*matrix_order+j;
        b_0x = _mm256_loadu_ps(b + offset_b); offset_b += matrix_order;
        b_1x = _mm256_loadu_ps(b + offset_b); offset_b += matrix_order;
        b_2x = _mm256_loadu_ps(b + offset_b); offset_b += matrix_order;
        b_3x = _mm256_loadu_ps(b + offset_b); offset_b += matrix_order;
        b_4x = _mm256_loadu_ps(b + offset_b); offset_b += matrix_order;
        b_5x = _mm256_loadu_ps(b + offset_b); offset_b += matrix_order;
        b_6x = _mm256_loadu_ps(b + offset_b); offset_b += matrix_order;
        b_7x = _mm256_loadu_ps(b + offset_b);

        for (int ii=0; ii<block_size; ++ii) {
          int offset_a = (i + ii) * matrix_order + k;
          a_x0 = _mm256_set1_ps(a[offset_a++]);
          a_x1 = _mm256_set1_ps(a[offset_a++]);
          a_x2 = _mm256_set1_ps(a[offset_a++]);
          a_x3 = _mm256_set1_ps(a[offset_a++]);
          a_x4 = _mm256_set1_ps(a[offset_a++]);
          a_x5 = _mm256_set1_ps(a[offset_a++]);
          a_x6 = _mm256_set1_ps(a[offset_a++]);
          a_x7 = _mm256_set1_ps(a[offset_a]);

          c[ii] = _mm256_add_ps(c[ii], _mm256_mul_ps(a_x0, b_0x));
          c[ii] = _mm256_add_ps(c[ii], _mm256_mul_ps(a_x1, b_1x));
          c[ii] = _mm256_add_ps(c[ii], _mm256_mul_ps(a_x2, b_2x));
          c[ii] = _mm256_add_ps(c[ii], _mm256_mul_ps(a_x3, b_3x));
          c[ii] = _mm256_add_ps(c[ii], _mm256_mul_ps(a_x4, b_4x));
          c[ii] = _mm256_add_ps(c[ii], _mm256_mul_ps(a_x5, b_5x));
          c[ii] = _mm256_add_ps(c[ii], _mm256_mul_ps(a_x6, b_6x));
          c[ii] = _mm256_add_ps(c[ii], _mm256_mul_ps(a_x7, b_7x));

        }

      }

      int offset_c = i*matrix_order+j;
      for (int iii=0; iii<block_size; ++iii) {
        _mm256_storeu_ps(result+offset_c, c[iii]);
        offset_c += matrix_order;
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
      result[block_size*i+j]
              = original_mtrx[original_mtrx_size*(i+top) + (j+left)];
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
      big_mtrx[big_mtrx_size*(i+top) + (j+left)] =
              block[block_size*i+j];
    }
  }
}




float* SquareMatrixMul_Blocked_OMP(float *a, float *b, int matrix_order, int block_size, int num_threads) {
  float *result = malloc(matrix_order*matrix_order * sizeof(*result));

  omp_set_num_threads(num_threads);
  #pragma omp parallel for collapse(2)
  for (size_t i=0; i<matrix_order; i+=block_size) {
    for (size_t j=0; j<matrix_order; j+=block_size) {
      float *block_result = InitZeroSquareMatrix(block_size);
      for (size_t k=0; k<matrix_order; k+=block_size) {
        float *block_a = SquareMatrixSelectSquareBlock(a,matrix_order,i,k,block_size);
        float *block_b = SquareMatrixSelectSquareBlock(b,matrix_order,k,j,block_size);
        float *block_c = SquareMatrixMul_Serial(block_a, block_b, block_size);
        SquareMatrixAddTo(block_result, block_c, block_size);
        free(block_a);
        free(block_b);
        free(block_c);
      }
      FillSquareBlockInToBigSquareMatrix(block_result,block_size,result,matrix_order,i,j);
      free(block_result);
    }
  }

  return result;
}


/**
 * @param a
 * @param b
 * @param matrix_order : must be multiple of 4
 * @return
 *
 * C_ij = sum(A_ik * B_kj)
 */
float* SquareMatrixMul_4x4Blocked_SSE_OMP(float *a, float *b, int matrix_order, int num_threads) {

  const int block_size = 4;

  float *result = malloc(matrix_order*matrix_order * sizeof(*result));

  omp_set_num_threads(num_threads);
  #pragma omp parallel for collapse(2)
  for (int i=0; i<matrix_order; i+=block_size) {
    for (int j=0; j<matrix_order; j+=block_size) {

      __m128 c[block_size];
      memset(c, 0, block_size*sizeof(*c));

      for (int k=0; k<matrix_order; k+=block_size) {
        __m128 a_x0,a_x1,a_x2,a_x3,
                b_0x,b_1x,b_2x,b_3x;

        int offset_b = k*matrix_order+j;
        b_0x = _mm_load_ps(b + offset_b); offset_b += matrix_order;
        b_1x = _mm_load_ps(b + offset_b); offset_b += matrix_order;
        b_2x = _mm_load_ps(b + offset_b); offset_b += matrix_order;
        b_3x = _mm_load_ps(b + offset_b);

        for (int ii=0; ii<block_size; ++ii) {
          int offset_a = (i + ii) * matrix_order + k;
          a_x0 = _mm_set1_ps(a[offset_a++]);
          a_x1 = _mm_set1_ps(a[offset_a++]);
          a_x2 = _mm_set1_ps(a[offset_a++]);
          a_x3 = _mm_set1_ps(a[offset_a]);

          c[ii] = _mm_add_ps(c[ii], _mm_mul_ps(a_x0, b_0x));
          c[ii] = _mm_add_ps(c[ii], _mm_mul_ps(a_x1, b_1x));
          c[ii] = _mm_add_ps(c[ii], _mm_mul_ps(a_x2, b_2x));
          c[ii] = _mm_add_ps(c[ii], _mm_mul_ps(a_x3, b_3x));

        }
      }

      int offset_c = i*matrix_order+j;
      for (int iii=0; iii<block_size; ++iii) {
        _mm_storeu_ps(result+offset_c, c[iii]);
        offset_c += matrix_order;
      }
    }
  }

  return result;
}

#endif //MATRIX_MULTIPLICATION_ACCELERATION_MATRIXCALCU_H

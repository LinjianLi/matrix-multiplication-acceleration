#cmake_minimum_required(VERSION 3.5)
project(matrix_multiplication_acceleration C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -mavx")

add_executable(MtrxMul_Serial main_simple_serial.c)
add_executable(MtrxMul_OMP main_simple_OMP.c)
add_executable(MtrxMul_Blocked_OMP main_Blocked_OMP.c)
add_executable(MtrxMul_SIMD_OMP main_SIMD_OMP.c)

add_executable(Matrix_mul main_Matrix_mul.c)
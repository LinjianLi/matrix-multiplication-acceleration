cmake_minimum_required(VERSION 3.13)
project(matrix_multiplication_acceleration C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -mavx")

add_executable(Matrix_mul main.c)
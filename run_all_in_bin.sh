cd bin

./Matrix_mul_simple_serial
./Matrix_mul_simple_OMP 8192 0.3
./Matrix_mul_Blocked_OMP 8192 0.3 12
./Matrix_mul_SIMD_OMP 8192 0.3 12

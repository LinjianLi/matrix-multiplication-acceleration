cd bin

./MtrxMul_Serial 8192 0.3
./MtrxMul_OMP 8192 0.3 32 0 -2
./MtrxMul_Blocked_OMP 8192 0.3 24
./MtrxMul_SIMD_OMP 8192 0.3 24

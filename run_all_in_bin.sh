cd bin

./MtrxMul_Serial 4096 0.3
./MtrxMul_OMP 4096 0.3 2 24 2
./MtrxMul_Blocked_OMP 4096 0.3 24
./MtrxMul_SIMD_OMP 4096 0.3 24

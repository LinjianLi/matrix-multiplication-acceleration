cd bin

./MtrxMul_Serial 4096 0.3
./MtrxMul_OMP 4096 0.3
./MtrxMul_Blocked_OMP 4096 0.3 12
./MtrxMul_SIMD_OMP 4096 0.3 12

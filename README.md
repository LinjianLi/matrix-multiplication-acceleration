# matrix-multiplication-acceleration

## Performance

CPU: XEON E2620, 2.00GHZ, 2 sockets, 6 cores per socket, 2 threads per core

OS： CentOS release 6.10

GCC: gcc 7.1.0

Matrix size: 8192x8192 float

Runtime:
* Serial: 34089s
* OMP 24 threads: 1972s
* 128x128 Blocked + OMP 24 Threads: 366
* SIMD:
  * 4x4 Blocked + SSE + OMP 24 Threads: 186s
  * 8x8 Blocked + AVX + OMP 24 Threads: 88s
* Strassen: unfinished


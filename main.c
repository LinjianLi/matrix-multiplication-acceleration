#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <locale.h>
#include "GenMatrix.h"
#include "MatrixCalcu.h"


int main(int argc, char *argv[]) {


  int n = (int) strtol(argv[1], NULL, 10);
  float seed = strtof(argv[2], NULL);
  float *a = malloc(n*n * sizeof(*a));  // float *a = new float[n*n];
  float *b = malloc(n*n * sizeof(*b));


  matrix_gen(a,b,n,seed);

//  printf("-----matrix a-----\n");// todo: delete
//  PrintSquareMatrix(a,n);// todo: delete
//  printf("-----matrix b-----\n");// todo: delete
//  PrintSquareMatrix(b,n);// todo: delete
//  printf("----------\n");// todo: delete

  float *c;

  int s;
  printf("Select a method: ");
  scanf("%d", &s);

  while (s>0) {
    // To record the start and end time of the multiplication.
    struct timeval start;
    struct timeval end;
    unsigned long diff;
    gettimeofday(&start,NULL);


    switch(s) {
      case 1:
        c = SquareMatrixMul_SimpleSerial(a,b,n);         // N: 1024. seed: 0.3. Trace: 252535. Time: 32602742ms.
        break;
      case 2:
        c = SquareMatrixMul_MultiThreadsByOMP(a,b,n);    // N: 1024. seed: 0.3. Trace: 252535. Time:  10517104ms.
        break;
      case 3:
        c = SquareMatrixMul_SplitToBlocks(a,b,n,16);     // N: 1024. seed: 0.3. Trace: 252535.  Time: 23690175ms.
      case 4:
        c = SquareMatrixMul_SplitToBlocks(a,b,n,32);     // N: 1024. seed: 0.3. Trace: 252535.  Time: 11215231ms.
        break;
      case 5:
        c = SquareMatrixMul_SplitToBlocks(a,b,n,64);     // N: 1024. seed: 0.3. Trace: 252535.  Time: 11036118ms.
        break;
      case 6:
        c = SquareMatrixMul_SplitToBlocks(a,b,n,128);    // N: 1024. seed: 0.3. Trace: 252535.  Time: 12629315ms.
        break;
      case 7:
        c = SquareMatrixMul_SplitToBlocks_MultiThreadsByOMP(a,b,n,16); // N: 1024. seed: 0.3. Trace: 252535.  Time: 10057319ms.
        break;
      case 8:
        c = SquareMatrixMul_SplitToBlocks_MultiThreadsByOMP(a,b,n,32); // N: 1024. seed: 0.3. Trace: 252535.  Time: 4469068ms.
        break;
      case 9:
        c = SquareMatrixMul_SplitToBlocks_MultiThreadsByOMP(a,b,n,64); // N: 1024. seed: 0.3. Trace: 252535.  Time: 3720381ms.
        break;
      case 10:
        c = SquareMatrixMul_SplitToBlocks_MultiThreadsByOMP(a,b,n,128); // N: 1024. seed: 0.3. Trace: 252535.  Time: 3736721ms.
        break;
      case 11:
        c = SquareMatrixMul_SSE(a,b,n);                                 // N: 1024. seed: 0.3. Trace: 252535.  Time: 3736721ms.
        break;
      case 12:
        c = SquareMatrixMul_AVX(a,b,n);                                 // N: 1024. seed: 0.3. Trace: 252535.  Time: 6086102ms.
        break;
      case 13:
        c = SquareMatrixMul_BlocksAndSSE(a,b,n,64);                    // N: 1024. seed: 0.3. Trace: 252535.  Time: 3736721ms.
        break;
      default:
        printf("Invalid input!\n");
    }


//    printf("-----matrix c-----\n");// todo: delete
//    PrintSquareMatrix(c,n);  // todo: delete
//    printf("----------\n");// todo: delete

    float trace = SquareMatrixTrace(c,n);
    printf("Trace %f\n", trace);


    // Calculate the time spent by the multiplication.
    gettimeofday(&end,NULL);
    diff = 1000000*(end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec);
    setlocale(LC_NUMERIC, "");
    printf("The time spent is %'ld microseconds\n----------\n", diff);
    // diff is time spent by the program, and the unit is microsecond


    printf("Select a method: ");
    scanf("%d", &s);
  }


  return 0;
}
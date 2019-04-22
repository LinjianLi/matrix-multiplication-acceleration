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


  FILE *log_file, *runtime_control_file;
  log_file = fopen("program-log.txt", "w");
  runtime_control_file = fopen("runtime-control.txt", "r");

  fprintf(log_file, "Order of matrix: %d\nSeed: %f\n----------\n", n, seed);

//  printf("-----matrix a-----\n");// todo: delete
//  PrintSquareMatrix(a,n);// todo: delete
//  printf("-----matrix b-----\n");// todo: delete
//  PrintSquareMatrix(b,n);// todo: delete
//  printf("----------\n");// todo: delete

  float *c;

  int s;
  //printf("Select a method: ");
//  scanf("%d", &s);
  fscanf(runtime_control_file, "%d", &s);

  while (s>0) {
    // To record the start and end time of the multiplication.
    struct timeval start;
    struct timeval end;
    unsigned long diff;
    gettimeofday(&start,NULL);


    switch(s) {
      case 1:
        c = SquareMatrixMul_SimpleSerial(a,b,n);
        break;
      case 2:
        c = SquareMatrixMul_MultiThreadsByOMP(a,b,n);
        break;
      case 3:
        c = SquareMatrixMul_SplitToBlocks(a,b,n,32);
        break;
      case 4:
        c = SquareMatrixMul_SplitToBlocks(a,b,n,64);
        break;
      case 5:
        c = SquareMatrixMul_SplitToBlocks(a,b,n,128);
        break;
      case 6:
        c = SquareMatrixMul_SplitToBlocks(a,b,n,256);
        break;
      case 7:
        c = SquareMatrixMul_SplitToBlocks_MultiThreadsByOMP(a,b,n,32);
        break;
      case 8:
        c = SquareMatrixMul_SplitToBlocks_MultiThreadsByOMP(a,b,n,64);
        break;
      case 9:
        c = SquareMatrixMul_SplitToBlocks_MultiThreadsByOMP(a,b,n,128);
        break;
      case 10:
        c = SquareMatrixMul_SplitToBlocks_MultiThreadsByOMP(a,b,n,256);
        break;
      case 11:
        c = SquareMatrixMul_SSE(a,b,n);
        break;
      case 12:
        c = SquareMatrixMul_AVX(a,b,n);
        break;
      case 13:
        c = SquareMatrixMul_BlocksAndSSE(a,b,n,64);
        break;
      default:
        fprintf(log_file, "Invalid input!\n");
        break;
    }

//    printf("-----matrix c-----\n");// todo: delete
//    PrintSquareMatrix(c,n);  // todo: delete
//    printf("----------\n");// todo: delete

    float trace = SquareMatrixTrace(c,n);

    fprintf(log_file, "method %d\tTrace: %f\n", s, trace);
//    printf("method %d\tTrace: %f\n", s, trace);

    // Calculate the time spent by the multiplication.
    gettimeofday(&end,NULL);
    diff = 1000000*(end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec);
    setlocale(LC_NUMERIC, "");
    fprintf(log_file, "The time spent is %'ld microseconds\n----------\n", diff);
    fflush(log_file);
//    printf("The time spent is %'ld microseconds\n----------\n", diff);
    // diff is time spent by the program, and the unit is microsecond


    free(c);

    //printf("Select a method: ");
//    scanf("%d", &s);
    fscanf(runtime_control_file, "%d", &s);
  }

  fclose(log_file);
  fclose(runtime_control_file);

  free(a);
  free(b);

  return 0;
}

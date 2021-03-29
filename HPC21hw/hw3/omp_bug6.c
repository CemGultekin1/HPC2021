/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
/*
Cem's Note:
Problems are multiple:
1-dotprod not returning a value somehow creates infinite loop
and program gets terminated
2-the writer wanted to use global variables however they are
re-initiated within the function scope

Solutions:
1-adding the return value sum
2-variables are all declared global and are not reinitiated within
the function scope
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];
int i;
float sum;
int tid;
float dotprod ()
{
  // int i,tid;
//float sum;
 tid = omp_get_thread_num();
#pragma omp for reduction(+:sum)
  for (i=0; i < VECLEN; i++)
    {
      sum  = sum + (a[i]*b[i]);
      printf("  tid= %d i=%d\n",tid,i);
    }
  return sum;
}


int main (int argc, char *argv[]) {

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

#pragma omp parallel shared(sum)
  dotprod();

printf("Sum = %f\n",sum);

}


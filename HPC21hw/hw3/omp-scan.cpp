#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  int nthreads;
  long *svec;
  long p;
  int tid;
#pragma omp parallel shared(nthreads,svec,p) private(tid)
  {
    tid=omp_get_thread_num();
    if(tid==0){
      nthreads = omp_get_num_threads();
      p = (long) ceil(((float)n)/((float)nthreads));
    }
  }
  printf("Number of CPUs %d\n",nthreads);
  #pragma omp parallel for
  for(long j=0;j<nthreads ; j++){
    long jp=((long) j)*p;
    long fin=((jp+p) > (n) ? (n) : (jp+p));
    if(j){prefix_sum[jp]=A[jp-1];}
    for (long i = jp+1; i <fin; i++) {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      }
  }
  svec=(long*) malloc((nthreads-1)*sizeof(long));
  svec[0]=prefix_sum[p-1];
  for(long i=1;i<nthreads-1;i++){
    svec[i]=svec[i-1]+prefix_sum[(i+1)*p-1];
  }
  
  #pragma omp parallel for
  for(long j=1;j<nthreads; j++){
    long jp=((long) j)*p;
    long fin=((jp+p) > (n) ? (n) : (jp+p));
    for (long i = jp; i <fin; i++) {
      prefix_sum[i] = prefix_sum[i] + svec[j-1];
    }
  }
}
int main() {
  long N =100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}

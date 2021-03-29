#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>


double gauss_seidel_iteration(double*u,double*f, int n){
  int i=0;
  int j=0;
  double h2=pow(1./((double) n),2.);
  int n2=pow(n,2);

#pragma omp parallel for private(i,j)
  for(int ii=0; ii<n2; ii++){
    j=ii%n;
    i=(int) floor(((float)(ii-j))/((float)n));
    if(i!=0 && i!=n-1 && j!=0 && j!=n-1 && (i+j) %2 ==1 ){
      //      printf(" (%d,%d)  i+j=%d  i+j%2=%d \n", i,j,i+j,i+j%2);
      u[ii]=1./4.*( u[ii-n] + u[ii+n] + u[ii+1] + u[ii-1] + h2*f[ii]);
    }
  }
#pragma omp parallel for private(i,j)
  for(int ii=0; ii<n2; ii++){
    j=ii%n;
    i=(int) floor(((float)(ii-j))/((float)n));
    if(i!=0 && i!=n-1 && j!=0 && j!=n-1 && (i+j) %2 ==0 ){
      //      printf(" (%d,%d) \n", i,j);

      u[ii]=1./4.*( u[ii-n] + u[ii+n] + u[ii+1] + u[ii-1] + h2*f[ii]);
    }
  }
  double res=0.;
#pragma omp parallel for reduction(+:res) private(i,j) 
  for(int ii=0; ii<n2; ii++){
    j=ii%n;
    i=(int) floor(((float)(ii-j))/((float)n));
    if(i!=0 && i!=n-1 && j!=0 && j!=n-1){
      res=res+pow(1./4.*( u[ii-n] + u[ii+n] + u[ii+1] + u[ii-1] + h2*f[ii])-u[ii],2.);
    }
  }

  return pow(res, 0.5);
}

int main(int argc, char *argv[]) {
  
  int N =atoi(argv[1]);
  int N2=pow(N,2);
  double* u = (double*) malloc(N2 * sizeof(double));
  double* f = (double*) malloc(N2 * sizeof(double));
  double h2=pow(1./((double) N),2);
  for (int ii = 0; ii < N2; ii++){
    f[ii] = 1.;
    u[ii] = 0.;
  }

  double tt = omp_get_wtime();
  int iternum=atoi(argv[2]);
  double* res = (double*) malloc((iternum+1) * sizeof(double));
  double initres=0;
  int i,j;



#pragma omp parallel for reduction(+:initres) private(i,j) 
  for(int ii=0; ii<N2; ii++){
    j=ii%N;
    i=(int) floor(((float)(ii-j))/((float)N));
    if(i!=0 && i!=N-1 && j!=0 && j!=N-1){
      initres=initres+pow(1./4.*( u[ii-N] + u[ii+N] + u[ii+1] + u[ii-1] + h2*f[ii])-u[ii],2.);
    }
  }
  res[0]=pow(initres,0.5);
  //  printf(" res[0]=%f\n",res[0]);
  
  for (int iter=1; iter<=iternum; iter++){
    res[iter]=gauss_seidel_iteration(u,f,N);
    //    printf(" res[%lu]=%f\n",iter,res[iter]);
  }
  printf("gauss-seidel: log10(res(%lu)/res(0)) = %f,   N=%lu, timing=%f, CPUnum=%d \n",	\
         iternum, log10(res[iternum]/res[0]),N, omp_get_wtime() - tt, atoi(argv[3]));
  free(u);
  free(f);
  return 0;
}

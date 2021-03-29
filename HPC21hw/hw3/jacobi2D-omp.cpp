#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>


double jacobi_iteration(double*u,double* v,double*f, long n){
  long i,j;
  double h2=pow(1./((double) n),2.);
  long n2=pow(n,2);

#pragma omp parallel for private(i,j)
  for(long ii=0; ii<n2; ii++){
    j=ii%n;
    i=(long) floor(((float)(ii-j))/((float)n));
    if(i!=0 && i!=n-1 && j!=0 && j!=n-1){
      v[ii]=1./4.*( u[ii-n] + u[ii+n] + u[ii+1] + u[ii-1] + h2*f[ii]);
    }
  }
  double res=0.;
#pragma omp parallel for reduction(+:res) private(i,j) 
  for(long ii=0; ii<n2; ii++){
    j=ii%n;
    i=(long) floor(((float)(ii-j))/((float)n));
    if(i!=0 && i!=n-1 && j!=0 && j!=n-1){
      res=res+pow(1./4.*( v[ii-n] + v[ii+n] + v[ii+1] + v[ii-1] + h2*f[ii])-v[ii],2.);
    }
    u[ii]=v[ii];
  }

  return pow(res,0.5);
}

int main(int argc, char* argv[]) {
  long N =atoi(argv[1]);
  long N2=pow(N,2);
  long n=N;
  long n2=N2;
  double* u = (double*) malloc(N2 * sizeof(double));
  double* v = (double*) malloc(N2 * sizeof(double));
  double* f = (double*) malloc(N2 * sizeof(double));
  double h2=pow(1./((double) N),2);
  long i;
  long j;
  for (long ii = 0; ii < N2; ii++){
    f[ii] = 1.;
    u[ii] = 0.;
    v[ii] = 0.;
  }
  double resinit=0.;
#pragma omp parallel for reduction(+:resinit) private(i,j) 
  for(long ii=0; ii<n2; ii++){
    j=ii%n;
    i=(long) floor(((float)(ii-j))/((float)n));
    if(i!=0 && i!=n-1 && j!=0 && j!=n-1){
      resinit=resinit+pow(1./4.*( v[ii-n] + v[ii+n] + v[ii+1] + v[ii-1] + h2*f[ii])-v[ii],2.);
    }
  }

  resinit= pow(resinit,0.5);

  
  double tt = omp_get_wtime();
  long iternum=atoi(argv[2]);
  double* res = (double*) malloc((iternum+1) * sizeof(double));
  res[0]=resinit;
  for (long iter=1; iter<=iternum; iter++){
    res[iter]=jacobi_iteration(u,v,f,N);
    //    printf(" res[%lu]=%f\n",iter,res[iter]);
  }

  printf("jacobi:  log10(res(%lu)/res(0)) = %f,   N=%lu, timing=%f, CPUnum=%d \n",\
	 iternum, log10(res[iternum]/res[0]),N, omp_get_wtime() - tt, atoi(argv[3]));
  free(u);
  free(v);
  free(f);
  return 0;
}

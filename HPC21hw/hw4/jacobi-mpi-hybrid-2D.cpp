/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq) {
  int i,j;
  double tmp, gres = 0.0, lres = 0.0;

#pragma omp parallel for default(none) shared(lu,lN,invhsq) private(i,j,tmp) reduction(+:lres)
  for (i = 1; i <= lN; i++){
    for(j = 1; j <= lN; j++){
      tmp = ((4.0*lu[i+(lN+2)*j] - lu[i-1+(lN+2)*j] - lu[i+1+(lN+2)*j] - lu[i+(lN+2)*(j-1)] - lu[i+(lN+2)*(j+1)]) * invhsq - 1);
      lres += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]) {
  int mpirank, i,j, p, N, lN, iter, max_iters;
  MPI_Status status, status1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);
# pragma omp parallel
  {
#ifdef _OPENMP
    int my_threadnum = omp_get_thread_num();
    int numthreads = omp_get_num_threads();
#else
    int my_threadnum = 0;
    int numthreads = 1;
#endif
    printf("Hello, I'm thread %d out of %d on mpirank %d\n", my_threadnum, numthreads, mpirank);
  }
  /* compute number of unknowns handled by each process */
  int p1 = floor(sqrt(p));
  
  lN = N / p1;
  if(mpirank==0)printf(" p1= %d\n",p1);
  int xrank= mpirank %p1;
  int yrank= mpirank/p1;
  if ((N % p1 != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
  double * inbndr = (double *) calloc(sizeof(double), lN*2);
  double * outbndr = (double *) calloc(sizeof(double), lN*2);
  double * lutemp;

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  
  gres0 = compute_residual(lu, lN, invhsq);
  if(mpirank==0) printf("residuals are computed\n");
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

#pragma omp parallel for default(none) shared(lN,lunew,lu,hsq) private(i,j)
    /* Jacobi step for local points */
    for (i = 1; i <= lN; i++){
      for (j = 1; j <= lN; j++){
        lunew[i + (lN+2)*j]  = 0.25 * (hsq + lu[i - 1 + (lN+2)*j] + lu[i + 1 + (lN+2)*j] + lu[i + (lN+2)*(j-1)] + lu[i + (lN+2)*(j+1)]);
      } 
    }

    for (i = 1; i <= lN; i++){
      inbndr[i-1]=lunew[(lN+2)*i+1];
    }
    for (i = 1; i <= lN; i++){
      inbndr[lN+i-1]=lunew[lN+(lN+2)*i];
    }
    //    printf("rank %d,%d has finished interior\n",xrank,yrank);
    if(xrank<p1 -1){
      //      printf(" (%d,%d) --> %d\n",xrank,yrank,mpirank+1); 
      MPI_Send(&(inbndr[lN]), lN, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
      //      printf(" (%d,%d) <-- %d\n",xrank,yrank,mpirank+1);
      MPI_Recv(&(outbndr[lN]), lN, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status);
      //      printf(" (%d,%d) communication is done",xrank,yrank);
      for (i = 1; i <= lN; i++){
	lunew[lN+1+(lN+2)*i]=outbndr[lN+i-1];
      }
    }
    if (xrank > 0) {
      /* If not the first process, send/recv bdry values to the left */
      //      printf(" (%d,%d) --> %d\n",xrank,yrank,mpirank-1);
      MPI_Send(&(inbndr[0]), lN, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      //      printf(" (%d,%d) <-- %d\n",xrank,yrank,mpirank-1);
      MPI_Recv(&(outbndr[0]), lN, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      //      printf(" (%d,%d) communication is done",xrank,yrank);
      for (i = 1; i <= lN; i++){
        lunew[(lN+2)*i]=outbndr[i-1];
      }
    }
    if(yrank<p1 -1){
      MPI_Send(&(lunew[(lN+2)*lN+1]), lN, MPI_DOUBLE, mpirank+p1, 124, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[(lN+2)*(lN+1)+1]), lN, MPI_DOUBLE, mpirank+p1, 123, MPI_COMM_WORLD, &status);
    }
    if (yrank > 0) {
      MPI_Send(&(lunew[lN+3]), lN, MPI_DOUBLE, mpirank-p1, 123, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[0]), lN, MPI_DOUBLE, mpirank-p1, 124, MPI_COMM_WORLD, &status);
    }

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);
  free(inbndr);
  free(outbndr);
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}

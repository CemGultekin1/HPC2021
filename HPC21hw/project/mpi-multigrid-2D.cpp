/* MPI-parallel multigrid Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * developed over Georg Stadler's code MPI-parllel Jacobi
 */
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <string.h>
#include <iostream>
#include <fstream>

using namespace std;

void ghost_node_comm(double*temp,double*lunew,int lN,int xrank,int yrank,int mpirank,int p1,double*stats){
  /*
    temp: needed extra memory for mpi receive and send
    lunew: target memory location
    lN: size of the local partitioning grid is lN+2 x lN+2
    xrank,yrank : coordinates of the subdomain across others
    p1: there are p1 x p1 many subdomains
    stats: collecting timings and values for analysis     
   */
  if(p1==1) return;
  int i;
  double tt0,tt1;
  MPI_Status status;
  double*inbndr=&(temp[0]);
  double*outbndr=&(temp[2*lN]);
  for (i = 0; i < lN+2; i++){
    inbndr[i]=lunew[(lN+2)*i+1];
  }
  for (i = 0; i < lN+2; i++){
    inbndr[lN+2+i]=lunew[lN+(lN+2)*i];
  }
  stats[2]+=4*lN;
  if(xrank<p1 -1){
    tt0=MPI_Wtime();
    MPI_Send(&(inbndr[lN+2]), lN+2, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
    MPI_Recv(&(outbndr[lN+2]), lN+2, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status);
    tt1=MPI_Wtime();
    stats[0]+=tt1-tt0;// comm. time
    stats[3]+=pow(lN+2,1)*2;//data acess
    for (i = 1; i <= lN; i++){
      lunew[lN+1+(lN+2)*i]=outbndr[lN+2+i];
    }
    stats[2]+=2*lN;// flops
  }
  if (xrank > 0) {
    tt0=MPI_Wtime();
    MPI_Send(&(inbndr[0]), lN+2, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
    MPI_Recv(&(outbndr[0]), lN+2, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status);
    tt1=MPI_Wtime();
    stats[0]+=tt1-tt0;
    stats[3]+=pow(lN+2,1)*2;
    for (i = 1; i <= lN; i++){
      lunew[(lN+2)*i]=outbndr[i];
    }
    stats[2]+=2*lN;
  }
  tt0=MPI_Wtime();
  if(yrank<p1 -1){
    MPI_Send(&(lunew[(lN+2)*lN]), lN+2, MPI_DOUBLE, mpirank+p1, 124, MPI_COMM_WORLD);
    MPI_Recv(&(lunew[(lN+2)*(lN+1)]), lN+2, MPI_DOUBLE, mpirank+p1, 123, MPI_COMM_WORLD, &status);
    stats[3]+=pow(lN+2,1)*2;
  }
  if (yrank > 0) {
    MPI_Send(&(lunew[lN+2]), lN+2, MPI_DOUBLE, mpirank-p1, 123, MPI_COMM_WORLD);
    MPI_Recv(&(lunew[0]), lN+2, MPI_DOUBLE, mpirank-p1, 124, MPI_COMM_WORLD, &status);
    stats[3]+=pow(lN+2,1)*2;    
  }
  tt1=MPI_Wtime();  
  stats[0]+=tt1-tt0;
}

double compute_residual(double *lu,double*f,double*f1, int lN, double invhsq,int p1,bool save_option,double*stats) {
  /*
    lu: partial solution
    f: the right hand side of the linear system
    f1: storage for new residue vector
    save_option: whether to store the computed residual values in f1 or not
   */
  double tt0,tt1;
  int i,j;
  double tmp, gres = 0.0, lres = 0.0;
  if(save_option){
#pragma omp parallel for default(none) shared(lu,f,f1,lN,invhsq) private(i,j,tmp) reduction(+:lres)
    for (i = 1; i <= lN; i++){
      for(j = 1; j <= lN; j++){
	tmp = ((4.0*lu[i+(lN+2)*j] - lu[i-1+(lN+2)*j] - lu[i+1+(lN+2)*j] - lu[i+(lN+2)*(j-1)] - lu[i+(lN+2)*(j+1)]) * invhsq - f[i+(lN+2)*j]);
	lres += tmp * tmp;
	f1[i+(lN+2)*j]=-tmp;
      }
    }
    stats[1]+=pow(lN,2)*9;
    stats[2]+=pow(lN,2)*7;

  }else{
#pragma omp parallel for default(none) shared(lu,f,lN,invhsq) private(i,j,tmp) reduction(+:lres)
    for (i = 1; i <= lN; i++){
      for(j = 1; j <= lN; j++){
	tmp = ((4.0*lu[i+(lN+2)*j] - lu[i-1+(lN+2)*j] - lu[i+1+(lN+2)*j] - lu[i+(lN+2)*(j-1)] - lu[i+(lN+2)*(j+1)]) * invhsq - f[i+(lN+2)*j]);
	lres += tmp * tmp;
      }  	
    }

    stats[1]+=pow(lN,2)*9;
    stats[2]+=pow(lN,2)*6;

  }
  tt0=MPI_Wtime();
  if(p1>1){
    MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }else{
    gres=lres;
  }
  tt1=MPI_Wtime();
  stats[0]+=tt1-tt0;
  return sqrt(gres);
}

void display(double*u, int lN){
  // for debugging purposes displaying matrices
  int i,j;
  for (i =0; i <lN+2; i++){
  for(j =0; j <lN+2; j++){
      printf("%g,  ",u[i+(lN+2)*j]);
    }  	
    printf("\n");
  }
  printf("\n\n\n");
}
double smooth(double*lf,double*lf1,double*lu,double*temp, int max_iters,int N,int lN, int p1,int mpirank,int xrank,int yrank,double*stats){// ofstream &myfile){
  /*
    Smooths the equation A * lu = lf and stores the new residue in lf1
    N is total grid size, lN is for this subdomain
   */
  int i,j,iter;
  double tt0,tt1;
  double*lunew=&(temp[0]);
  double*comm_temp=&(temp[(lN+2)*(lN+2)]);
  double omega=2./3.;
  for(i=0;i<(lN+2)*(lN+2);i++) lunew[i]=0;
  double* lutemp;
  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-6;
  bool streamopen=false;
  gres0 = compute_residual(lu, lf,lf1,lN,invhsq,p1,false,stats);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres> tol; iter++) {
#pragma omp parallel for default(none) shared(lN,lunew,lu,hsq,lf,omega) private(i,j)
    /* Jacobi step for local points */
    for (i = 1; i <= lN; i++){
      for (j = 1; j <= lN; j++){
        lunew[i + (lN+2)*j]  =lu[i + (lN+2)*j]+omega* 0.25 * (hsq*lf[i + (lN+2)*j] +
							      lu[i - 1 + (lN+2)*j] + lu[i + 1 + (lN+2)*j] + lu[i + (lN+2)*(j-1)] + lu[i + (lN+2)*(j+1)]
							      -4*lu[i + (lN+2)*j]);
      } 
    }
    stats[1]+=pow(lN,2)*6;
    stats[2]+=pow(lN,2)*6;
    ghost_node_comm(comm_temp,lunew,lN,xrank,yrank,mpirank,p1,stats);
    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lf,lf1,lN,invhsq,p1,false,stats);
    }
  }
  return compute_residual(lu, lf,lf1,lN,invhsq,p1,true,stats);
}

void downsample(double*u,double*temp,int lN,int p1,int mpirank,int xrank,int yrank,double*stats){
  /*
    Downsampling u from 2*lN grid to lN grid
   */
  int i,j,ii,jj,j1,jn1,imax,jmax,N;
 if(xrank<p1-1){
    imax=lN;
  }else{
    imax=lN-1;
  }
  if(yrank<p1-1){
    jmax=lN;
  }else{
    jmax=lN-1;
  }
  for (i = 1; i <= imax; i++){
    for (j = 1; j <= jmax; j++){
      ii=i + (lN+2)*j;
      jj=2*i+(2*lN+2)*2*j;
      j1=jj+(2*lN+2);
      jn1=jj-(2*lN+2);
      u[ii]  = ( 4*u[jj]+2*u[jj+1]+2*u[jj-1]+2*u[j1]+u[j1+1]+u[j1-1]+2*u[jn1]+u[jn1+1]+u[jn1-1])/16.;
    }
  }
  ghost_node_comm(temp,u, lN,xrank, yrank, mpirank, p1,stats);
}

void upsample_and_add(double*u1,double*u0,int lN,int p1,int mpirank,int xrank,int yrank,double*stats){
  /*
    upsampling u0 from lN to 2*lN grid and adding it to u1
   */
  int i,j,ii,jj,j1,jp1,ip1;
  double beta=0.1;
  for (i = 0; i < lN+1; i++){
    for (j = 0; j < lN+1; j++){
      ii=i + (lN+2)*j;
      ip1=ii + (lN+2);
      jj=2*i+(2*lN+2)*2*j;
      u1[jj]+=beta*u0[ii];
      jp1=jj+(2*lN+2);
      u1[jj+1]+=beta*0.5*(u0[ii]+u0[ii+1]);
      u1[jp1]+=beta*0.5*(u0[ii]+u0[ip1]);
      u1[jp1+1]+=beta*0.25*(u0[ii]+u0[ii+1]+u0[ip1]+u0[ip1+1]);
    }
  }
  stats[1]+=15*pow(lN+2,2);
  stats[2]+=14*pow(lN+2,2);
  if(xrank==p1-1){
    for (j = 0; j < 2*lN+2; j++){
       jj=2*lN+1+(2*lN+2)*j;
       u1[jj]=0;
    }
    stats[2]+=2*lN+2;
  }
  if(yrank==p1-1){
    for (i = 0; i < 2*lN+2; i++){
      jj=i+(2*lN+2)*(2*lN+1);
      u1[jj]=0;
    }
    stats[2]+=2*lN+2;
  }
}


double v_cycle(double*lf,double*lu,double*temp, int max_iters,int N,int lN,int depth,int p,int p1,int mpirank,int xrank,int yrank,double* stats){
  /*
    Single V-cycle solving A*lu=lf
    temp is for extra needed memory
    p1xp1=p many nodes exist in total
   */
  double tt0,tt1;
  tt0 = MPI_Wtime();
  
# pragma omp parallel
  {
#ifdef _OPENMP
    int my_threadnum = omp_get_thread_num();
    int numthreads = omp_get_num_threads();
#else
    int my_threadnum = 0;
    int numthreads = 1;
#endif
  }
 
  int J,lNJ,NJ,curloc,newloc,ss;
  int* locs=(int*) calloc(sizeof(int),depth);
  // Going down, coarsing
  for(J=1; J<depth;J++){
    lNJ=lN/pow(2,J-1);
    NJ=N/pow(2,J-1);
    curloc=locs[J-1];
    newloc=curloc+pow(lNJ+2,2);
    locs[J]=newloc;
    // lf[curloc] is the right hand side for the linear system, the residue is stored in lf[newloc]
    smooth(&(lf[curloc]),&(lf[newloc]),&(lu[curloc]),temp, max_iters, NJ,lNJ,p1,mpirank,xrank,yrank,stats);
    // lf[newloc] downsampled and stored on the same array. It is to be used in the next iteration.
    downsample(&(lf[newloc]),temp,lNJ/2,p1,mpirank,xrank,yrank,stats);
  }
  // Deepest
  J=depth;
  NJ=N/pow(2,J-1);
  curloc=locs[J-1];
  lNJ=lN/pow(2,J-1);
  newloc=curloc+pow(lNJ+2,2);
  double gres;
  gres=smooth(&(lf[curloc]),&(lf[newloc]),&(lu[curloc]),temp, max_iters, NJ,lNJ,p1,mpirank,xrank,yrank,stats);

  // Going up, refining
  for(J=depth-1; J>0;J--){
    lNJ=lN/pow(2,J-1);
    NJ=N/pow(2,J-1);
    curloc=locs[J-1];
    newloc=locs[J];
    upsample_and_add(&(lu[curloc]),&(lu[newloc]),lNJ/2,p1,mpirank,xrank,yrank,stats);
    gres=smooth(&(lf[curloc]),&(lf[newloc]),&(lu[curloc]),temp, max_iters, NJ,lNJ,p1,mpirank,xrank,yrank,stats);
  }
  if(mpirank==0) printf("\t\t gres=%e\n",gres);
  return gres;  
}


void experiment(int N,int mpirank,int p,int depth,int niter,double**  stats){
  /*
    Manages memory, performance recordings across multiple V-cycles
     for N+2 x N+2 grid divided into p subdomains with V-cycles of depth "depth"
   */
  double tt0,tt1;
  tt0 = MPI_Wtime();
  int p1,xrank,yrank,lN,max_iters;
  p1 = floor(sqrt(p));
  xrank= mpirank %p1;
  yrank= mpirank/p1;
  
  lN = N / p1;
  max_iters=1000;

  // Computing needed memory size foe V-cycle
  int lN_=lN;
  int tot_mem,i,j,k,l;
  tot_mem=0;
  for(i=0; i<depth+1;i++){
    tot_mem+= pow(lN_+2,2);
    lN_=lN_/2;
  }
  tot_mem+= pow(lN*2+2,2);
  // multiplying for 2 for safety. During V-cycle, large memory is temporarily put in places of smaller memory for speed
  tot_mem=tot_mem*2;
  
  double * lu    = (double *) calloc(sizeof(double), tot_mem);
  double * temp =  (double *) calloc(sizeof(double), tot_mem);
  double * lf =(double *) calloc(sizeof(double),tot_mem);

  for(i=0;i<lN + 2; i++){
    for(j=0;j<lN+2;j++){      
      lf[i+(lN+2)*j]=1;
    }
  }
  stats[0][2]+=pow(lN+2,2)*2;
  for(k=0;k<niter;k++){
    if(mpirank==0) printf("\t\t iter #%i\n",k);
    stats[k][4]+=v_cycle(lf,lu,temp,max_iters,N,lN,depth,p,p1,mpirank,xrank,yrank,stats[k]);
    MPI_Barrier(MPI_COMM_WORLD);
    tt1 = MPI_Wtime();
    stats[k][5]+=tt1-tt0;// total time of V-cycles collected cumulatively
    tt0 = tt1;
  }
  free(lu);
  free(temp);
  free(lf);

}
int main(int argc, char ** argv) {
  int  N,depth,nrepeats,nsizes,base_size,ndevice,i,j,k,l;
  int mpirank,p,p_;
  int nstat,niter;

  int *sizes;// number of N values to be tested
  int *depths;// storing various v-cycle depth values
  MPI_Status status;

  int ndepths=5;// number of depths to be tested
  depths=new int [ndepths];
  i=0;
  depths[i]=1;i++;
  depths[i]=2;i++;
  depths[i]=3;i++;
  depths[i]=4;i++;
  depths[i]=5;i++;

  nstat=6;
  niter=300;// number of V-cycle iterations
  nrepeats=1;
  nsizes=1;//16

  double** stats;// recording timings and flops and residues
  stats=new double* [niter];
  for(i=0;i<niter;i++){
    stats[i]=new double [nstat];
  }
  /*
    indices of stats are arranged as follows:
    0- communication time
    1- flops
    2- data access
    3- communication size
    4- residue
    5- compute time
  
  */

  double* stats_;// temporary memory needed for communicating stats
  stats_=new double [niter*nstat];

  // to store the recordings in a file
  string filename;
  ofstream myfile;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if(mpirank==0){
    filename="results-"+to_string(p)+"x.txt"; 
    myfile.open(filename);
  }

  
  base_size=((int) sqrt(p))*pow(2,6);
  sizes=new int [nsizes];
  for(i=0;i<nsizes;i++){
    sizes[i]=base_size*(i+1);
  }


  // storing variables for each experiment into a 2D array
  int EXP[nsizes*ndepths][3];
  for(j=0; j<nsizes*ndepths;j++){
    i=j%nsizes;
    k=(j-i)/nsizes;
    EXP[j][0]=sizes[i];
    EXP[j][1]=depths[k];
    EXP[j][2]=p;
   
  }
  
  for(l=0;l<nsizes*ndepths;l++){
    N=EXP[l][0];
    depth=EXP[l][1];
    p_=EXP[l][2];
    if(!mpirank) printf("#%i\t N=%i, depth=%i, p=%i\n",l,N,depth,p_);
    if( N> pow(2,depth)*sqrt(p_)){

      // initiating records to 0
      for(i=0;i<niter;i++){
	for(j=0;j<nstat;j++){
	  stats[i][j]=0.;
	}
      }

      // running the experiment
      for( i=0; i<nrepeats; i++){
	if(mpirank<p_){
	  experiment(N, mpirank,p_, depth,niter,stats);
	}
      }

      // averaging over nrepets
      for(i=0; i<niter; i++){
	for(j=0;j<nstat;j++){
	  stats[i][j]=stats[i][j]/( (double) nrepeats);
	}
	// subtracting from total time the communication time to find computation time
	stats[i][5]=stats[i][5]-stats[i][0];
      }

      // timings increase cumulatively across the iterations of v-cycle
      for(i=0;i<niter-1;i++){
	stats[i+1][0]+=stats[i][0];
	stats[i+1][1]+=stats[i][1];
	stats[i+1][2]+=stats[i][2];
	stats[i+1][3]+=stats[i][3];      
	stats[i+1][5]+=stats[i][5];
      }

      // collecting recordings from other nodes and then averaging them
      // not all nodes are the same. boundary subdomains have less communication
      if(mpirank!=0){
	for(i=0;i<niter;i++){
	  for(j=0;j<nstat;j++){
	    stats_[i*nstat+j]=stats[i][j];
	  }
	}
	MPI_Send(&(stats_[0]), nstat*niter, MPI_DOUBLE, 0, 120+(mpirank-1), MPI_COMM_WORLD);
      }else{
	for(j=1;j<p_;j++){
	  MPI_Recv(&(stats_[0]), nstat*niter, MPI_DOUBLE, j,  120+(j-1), MPI_COMM_WORLD, &status);
	  for(i=0;i<niter;i++){
	    for(k=0;k<nstat;k++){
	      stats[i][k]+=stats_[i*nstat+k];
	    }
	  }
	}
	for(i=0;i<niter;i++){
	  for(k=0;k<nstat;k++){
	    stats[i][k]=stats[i][k]/((double) p_);
	  }
	}
      }
      
      if (0 == mpirank) {
	//printing some results
	printf("N=%i\n",N);
	printf("communication time :\t\t %g s\n",stats[0][0]);    
	printf("compute time:\t\t %g s\n",stats[0][5]);
	printf("flops per sec:\t\t %g Gf/s\n",stats[0][1]/stats[0][5]/1e9);
	printf("server bandwidth:\t\t %g GB/s\n",8*stats[0][3]/stats[0][0]/1e9);
	printf("internal bandwidth:\t\t %g GB/s\n",8*stats[0][2]/stats[0][5]/1e9);
	for(i=0;i<niter;i++){
	  //writing to file the results
	  myfile<<"N= "<<N<<",\t depth= "<<depth<<",\t p= "<<p_<<",\t niter= "<<i+1<<endl;
	  myfile<<"flops per sec \t\t" << stats[i][1]/stats[i][5]/1e9<<endl;
	  myfile<<"internal bandwidth \t" << 8*stats[i][2]/stats[i][5]/1e9<<endl;
	  myfile<<"server bandwidth\t"<<8*stats[i][3]/stats[i][0]/1e9<<endl;
	  myfile<<"residue \t\t"<<stats[i][4]<<endl;
	  myfile<<"compute time \t\t"<<stats[i][5]<<endl;
	  myfile<<"comm time \t\t"<<stats[i][0]<<endl;
	}
      }
    }
  }

  
  for(i=0;i<niter;i++){
    free(stats[i]);
  }
  free(stats);
  if(mpirank==0){
    myfile.close();
  }
  MPI_Finalize();
  return 0;
}

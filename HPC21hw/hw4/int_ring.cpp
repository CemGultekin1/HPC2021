/* Simple send-receive example */
#include <stdio.h>
#include <mpi.h>
double time_ring(int Nsize, int Nrepeats){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int* msg = (int*) calloc(sizeof(int), Nsize);
  msg[0]=Nrepeats-1;
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();
  MPI_Status status;
  int  next = (rank + 1) % 3;
  int prev = (rank + 2) % 3;
  if (rank == 0) {
    MPI_Send(&(msg[0]), Nsize, MPI_INT, next, 999, MPI_COMM_WORLD);
  }
  while(1){
    if(rank<3){
      MPI_Recv(&(msg[0]), Nsize, MPI_INT, prev, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if(rank==0){
        msg[0]-=1;
      }
      msg[1]=msg[1]+rank;
      
      
      MPI_Send(&(msg[0]), Nsize, MPI_INT, next, 999, MPI_COMM_WORLD);
      if(rank==2){
      }
      if(msg[0]==0){
	break;
      }
    }
  }
  if (0 == rank) {
    MPI_Recv(msg, Nsize, MPI_INT, prev, 999, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    printf("the result= %d\n", msg[1]);}
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0){
    tt = MPI_Wtime()-tt;
  }
  free(msg);
  return tt;
}
int main(int argc, char *argv[]) {
  int rank;
  MPI_Init(&argc, &argv);
  int Nsize=2000000/4;
  int Nrepeats=50;
  double tt=time_ring(2,Nrepeats);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if(!rank) printf("latency %e\n",tt);

  MPI_Barrier(MPI_COMM_WORLD);
  tt=time_ring(Nsize,Nrepeats);
  if(!rank)  printf("bandwidth  %e GB/s\n", (Nsize*Nrepeats*4)/tt/1e9);
  MPI_Finalize();
  return 0;
}

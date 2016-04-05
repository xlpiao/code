#include <stdio.h>
#include <stdlib.h>

__global__ void colonel(int *a_d){
  atomicAdd( a_d, blockIdx.x * blockDim.x + threadIdx.x);
}

int main(){

  int a = 0, *a_d;
  
  cudaMalloc((void**) &a_d, sizeof(int));
  cudaMemcpy(a_d, &a, sizeof(int), cudaMemcpyHostToDevice);

  float   elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );

  colonel<<<4,4>>>(a_d); // global id: 0 ~ 15, atomicAdd = sum(0+1+2+3+...+15) 
  
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &elapsedTime, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  printf("GPU Time elapsed: %f seconds\n", elapsedTime/1000.0);
  
  
  cudaMemcpy(&a, a_d, sizeof(int), cudaMemcpyDeviceToHost);

  printf("a = %d\n", a);
  cudaFree(a_d);

}

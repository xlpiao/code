#include <stdlib.h>
#include <stdio.h>

__device__ int get_global_index(void)
{
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_constant(void)
{
  return 7;
}

__global__ void kernel1(int *array)
{
  int index = get_global_index();
  array[index] = get_constant();
}

__global__ void kernel2(int *array)
{
  int index = get_global_index();
  array[index] = get_global_index();
}

int main(void)
{
  int num_elements = 256;
  int num_bytes = num_elements * sizeof(int);

  int *device_array = 0;
  int *host_array = 0;

  // allocate memory
  host_array = (int*)malloc(num_bytes);
  cudaMalloc((void**)&device_array, num_bytes);

  int block_size = 16;
  int grid_size = num_elements / block_size;

  // launch kernel1 and inspect its results
  kernel1<<<grid_size,block_size>>>(device_array);
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  printf("kernel1 results:\n");
  for(int i = 0; i < num_elements; ++i)
  {
    printf("%3d ", host_array[i]);
    if((i+1) % block_size == 0)
        printf("\n");
  }
  printf("\n");

  // launch kernel2 and inspect its results
  kernel2<<<grid_size,block_size>>>(device_array);
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  printf("kernel2 results:\n");
  for(int i = 0; i < num_elements; ++i)
  {
    printf("%3d ", host_array[i]);
    if((i+1) % block_size == 0)
        printf("\n");
  }
  printf("\n");

  // deallocate memory
  free(host_array);
  cudaFree(device_array);
  return 0;
}

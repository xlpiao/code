#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int num_elements = 16;
  int num_bytes = num_elements * sizeof(int);

  int *device_array = 0;
  int *host_array = 0;

  // malloc host memory
  host_array = (int *)malloc(num_bytes);

  // cudaMalloc device memory
  cudaMalloc((void **)&device_array, num_bytes);

  // zero out the device array with cudaMemset
  cudaMemset(device_array, 0, num_bytes);

  // copy the contents of the device array to the host
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // print out the result element by element
  for (int i = 0; i < num_elements; ++i) printf("%d ", host_array[i]);
  printf("\n");

  // use free to deallocate the host array
  free(host_array);

  // use cudaFree to deallocate the device array
  cudaFree(device_array);

  return 0;
}

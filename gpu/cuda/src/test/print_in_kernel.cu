// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

__global__ void testKernel(int val) {
  cuPrintf("Hello, world from the device!\n");
  cuPrintf("[%d, %d]:\t\tValue is:%d\n", blockIdx.y * gridDim.x + blockIdx.x,
           threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
               threadIdx.x,
           val);
}

int main(int argc, char **argv) {
  int devID;
  cudaDeviceProp props;

  printf("\n*****Hello, world from the host!*****\n\n");

  // This will pick the best possible CUDA capable device
  devID = findCudaDevice(argc, (const char **)argv);

  // Get GPU information
  checkCudaErrors(cudaGetDevice(&devID));
  checkCudaErrors(cudaGetDeviceProperties(&props, devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
         props.major, props.minor);

  printf("printf() is called. Output:\n\n");

  // Kernel configuration, where a two-dimensional grid and
  // three-dimensional blocks are configured.
  dim3 dimGrid(2, 2);
  dim3 dimBlock(2, 2, 2);
  testKernel<<<dimGrid, dimBlock>>>(10);
  cudaDeviceSynchronize();

  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaDeviceReset();

  return EXIT_SUCCESS;
}

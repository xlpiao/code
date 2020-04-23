#include <stdio.h>
#include <iostream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main() {
  cl_int err;

  // get first platform
  cl_platform_id platform;
  err = clGetPlatformIDs(1, &platform, NULL);

  // get device count
  cl_uint deviceCount;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
  printf("deviceCount: %d\n", deviceCount);

  // get all devices
  cl_device_id* devices;
  devices = new cl_device_id[deviceCount];
  err =
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

  // for each device create a separate context AND queue
  cl_context* contexts = new cl_context[deviceCount];
  cl_command_queue* queues = new cl_command_queue[deviceCount];
  for (int i = 0; i < deviceCount; i++) {
    char stringOfDevice[1024];
    err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(stringOfDevice),
                          &stringOfDevice, NULL);
    printf("context and queue id: %d, device_name: %s\n", i, stringOfDevice);
    contexts[i] = clCreateContext(NULL, deviceCount, devices, NULL, NULL, &err);
    queues[i] = clCreateCommandQueue(contexts[i], devices[i], 0, &err);
  }

  /*
   * Here you have one context and one command queue per device.
   * You can choose to send your tasks to any of these queues.
   */

  // cleanup
  for (int i = 0; i < deviceCount; i++) {
    clReleaseDevice(devices[i]);
    clReleaseContext(contexts[i]);
    clReleaseCommandQueue(queues[i]);
  }

  delete[] devices;
  delete[] contexts;
  delete[] queues;

  return 0;
}

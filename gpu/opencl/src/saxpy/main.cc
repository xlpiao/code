/*
 * Copyright (c) 2016 Xianglan Piao <xianglan0502@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "util.h"

#define N 512

void printArray(float *arr) {
  int correct = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", arr[i * N + j]);
    }
  }
  printf("\n");
}

#define TOL 0.001
void correctnessCheck(float alpha, float *h_x, float *h_y, float *h_z) {
  int correct = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float temp = alpha * h_x[i * N + j] + h_y[i * N + j];
      temp -= h_z[i * N + j];
      if (temp * temp < TOL * TOL) {
        correct++;
      }
    }
  }
  printf("\tCheck Correctness: %d  /  %d\n\n", correct, N * N);
}

void computeInDevice(cl_device_id &device_id) {
  //// data initialization
  float *h_x = (float *)malloc(N * N * sizeof(float));  // NxN
  float *h_y = (float *)malloc(N * N * sizeof(float));  // NxN
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      h_x[i * N + j] = 1;
      h_y[i * N + j] = 1;
    }
  }

  //// opencl operation
  cl_int err = CL_SUCCESS;

  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

  cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * N * N,
                              NULL, NULL);
  cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * N * N,
                              NULL, NULL);
  cl_mem d_z = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N * N,
                              NULL, NULL);

  cl_command_queue cmd_queue =
      clCreateCommandQueue(context, device_id, 0, &err);

  err = clEnqueueWriteBuffer(cmd_queue, d_x, CL_TRUE, 0, sizeof(float) * N * N,
                             h_x, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(cmd_queue, d_y, CL_TRUE, 0, sizeof(float) * N * N,
                             h_y, 0, NULL, NULL);

  std::string::size_type pos = std::string(__FILE__).find_last_of("\\/");
  auto kernel_path = std::string(__FILE__).substr(0, pos).append("/saxpy.cl");
  // std::cout << kernel_path << std::endl;
  std::ifstream file(kernel_path);

  std::ostringstream out;
  out << file.rdbuf();
  std::string kernel_src = out.str();
  auto kernel_str = (char *)kernel_src.c_str();
  auto kernel_str_size = kernel_src.size();
  // std::cout << kernel_str << std::endl;

  cl_program program =
      clCreateProgramWithSource(context, 1, (const char **)&kernel_str,
                                (const size_t *)&kernel_str_size, &err);

  err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  cl_kernel kernel = clCreateKernel(program, "saxpy", &err);

  const float alpha = 1.0f;
  int n = N;
  err = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_x);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_y);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_z);
  err = clSetKernelArg(kernel, 4, sizeof(int), (void *)&n);

  size_t local_work_group_size;
  err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(local_work_group_size),
                                 &local_work_group_size, NULL);

  size_t localWorkSize[2] = {local_work_group_size, 1};
  size_t globalWorkSize[2] = {N, N};

  std::cout << "\tLocalWorkSize: " << localWorkSize[0] << ", "
            << localWorkSize[1] << std::endl;
  std::cout << "\tGlobalWorkSize: " << globalWorkSize[0] << ", "
            << globalWorkSize[1] << std::endl;

  auto time_start = std::chrono::high_resolution_clock::now();

  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, globalWorkSize,
                               localWorkSize, 0, NULL, NULL);
  clFinish(cmd_queue);

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                       time_end - time_start)
                       .count();
  std::cout << "\033[0;31m\tExe. Time: " << time_diff << " ms\033[0m"
            << std::endl;

  //// correctness check
  float *h_z = (float *)malloc(N * N * sizeof(float));
  err = clEnqueueReadBuffer(cmd_queue, d_z, CL_TRUE, 0, N * N * sizeof(float),
                            h_z, 0, NULL, NULL);
  correctnessCheck(alpha, h_x, h_y, h_z);
  free(h_z);

  //// memory release
  err = clFlush(cmd_queue);
  err = clReleaseKernel(kernel);
  err = clReleaseProgram(program);
  err = clReleaseMemObject(d_x);
  err = clReleaseMemObject(d_y);
  err = clReleaseMemObject(d_z);
  err = clReleaseCommandQueue(cmd_queue);
  err = clReleaseContext(context);

  free(h_x);
  free(h_y);
}

int main(void) {
  cl_int err = CL_SUCCESS;

  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  std::cout << "\033[0;31m\nNumber of Platforms: \033[0m" << num_platforms
            << std::endl;

  cl_platform_id *platform_ids = new cl_platform_id[num_platforms];
  err = clGetPlatformIDs(num_platforms, platform_ids, NULL);

  for (int i = 0; i < num_platforms; i++) {
    std::cout << "\033[0;31m[Platform #" << i << "]\033[0m" << std::endl;
    // util::printPlatformInfo(platform_ids[i]);

    cl_uint num_devices;
    err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                         &num_devices);
    std::cout << "\033[0;31m\n\tNumber of Devices: \033[0m" << num_devices
              << std::endl;

    cl_device_id *device_ids = new cl_device_id[num_devices];
    err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, num_devices,
                         device_ids, &num_devices);
    for (int j = 0; j < num_devices; j++) {
      std::cout << "\033[0;31m\t[Platform #" << i << "][Device #" << j
                << "]\033[0m" << std::endl;
      // util::printDeviceInfo(device_ids[j]);

      cl_device_type type;
      clGetDeviceInfo(device_ids[j], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
      if (type & CL_DEVICE_TYPE_CPU) {
        std::cout << "\tDevice Type: CL_DEVICE_TYPE_CPU" << std::endl;
        std::thread cpu_thread(computeInDevice, std::ref(device_ids[j]));
        cpu_thread.join();
      } else if (type & CL_DEVICE_TYPE_GPU) {
        std::cout << "\tDevice Type: CL_DEVICE_TYPE_GPU" << std::endl;
        std::thread gpu_thread(computeInDevice, std::ref(device_ids[j]));
        gpu_thread.join();
      } else if (type & CL_DEVICE_TYPE_ACCELERATOR) {
        std::cout << "\tDevice Type: CL_DEVICE_TYPE_ACCELERATOR" << std::endl;
      } else if (type & CL_DEVICE_TYPE_DEFAULT) {
        std::cout << "\tDevice Type: CL_DEVICE_TYPE_DEFAULT" << std::endl;
      }
    }
    delete[] device_ids;
  }
  delete[] platform_ids;

  return 0;
}

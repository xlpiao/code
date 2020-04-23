/*
 * Gemm.cc
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
#include <filesystem>
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

#define LOCAL_WORK_GROUP_X 16
#define LOCAL_WORK_GROUP_Y 1
#define ALPHA 1
#define BETA 1
#define N 512

float *in_a;
float *in_b;
float *out_parallel;
float *out_sequential;

void printArray(float *out_parallel) {
  int correct = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", out_parallel[i * N + j]);
    }
  }
  printf("\n");
}

void correctnessCheck(float *in_a, float *in_b) {
  int correct = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (in_a[i * N + j] == in_b[i * N + j]) {
        correct++;
      }
    }
  }
  printf("\tCheck Correctness: %d  /  %d\n\n", correct, N * N);
}

void init(float *in_a, float *in_b, float *out_parallel) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      in_a[i * N + j] = 1;
      in_b[i * N + j] = 1;
      // out_parallel[i*N + j] = 0;
    }
  }
}

void seqGemm(void) {
  auto time_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      out_sequential[i * N + j] *= BETA;

      for (int k = 0; k < N; ++k) {
        out_sequential[i * N + j] += ALPHA * in_a[i * N + k] * in_b[k * N + j];
      }
    }
  }
  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                       time_end - time_start)
                       .count();
  // printArray(out_sequential);
  std::cout << "\n\033[0;31mCPU (Seq GEMM)\n\tExe. Time: " << time_diff
            << " ms\033[0m" << std::endl;
}

void computeInDevice(cl_device_id &device_id) {
  cl_int err = CL_SUCCESS;

  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

  int n = N;
  float alpha = ALPHA;
  float beta = BETA;
  cl_mem in_a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * N * N, NULL, NULL);
  cl_mem in_b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * N * N, NULL, NULL);
  cl_mem out_parallel_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           sizeof(float) * N * N, NULL, NULL);

  cl_command_queue cmd_queue =
      clCreateCommandQueue(context, device_id, 0, &err);

  err = clEnqueueWriteBuffer(cmd_queue, in_a_mem, CL_TRUE, 0,
                             sizeof(float) * N * N, in_a, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(cmd_queue, in_b_mem, CL_TRUE, 0,
                             sizeof(float) * N * N, in_b, 0, NULL, NULL);
  err =
      clEnqueueWriteBuffer(cmd_queue, out_parallel_mem, CL_TRUE, 0,
                           sizeof(float) * N * N, out_parallel, 0, NULL, NULL);

  auto pwd = std::filesystem::current_path();
  std::ifstream file(pwd.append("../Gemm/gemm.cl"));
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

  cl_kernel kernel = clCreateKernel(program, "gemm", &err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_a_mem);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&in_b_mem);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&out_parallel_mem);
  err |= clSetKernelArg(kernel, 3, sizeof(float), (void *)&alpha);
  err |= clSetKernelArg(kernel, 4, sizeof(float), (void *)&beta);
  err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&n);

  size_t localWorkSize[2] = {LOCAL_WORK_GROUP_X, LOCAL_WORK_GROUP_Y};
  size_t globalWorkSize[2] = {
      (size_t)std::ceil((float)N / (float)LOCAL_WORK_GROUP_X) *
          LOCAL_WORK_GROUP_X,
      (size_t)std::ceil(((float)N) / ((float)LOCAL_WORK_GROUP_Y)) *
          LOCAL_WORK_GROUP_Y};
  std::cout << "\tLocalWorkSize: " << localWorkSize[0] << ", "
            << localWorkSize[1] << std::endl;
  std::cout << "\tGlobalWorkSize: " << globalWorkSize[0] << ", "
            << globalWorkSize[1] << std::endl;

  auto time_start = std::chrono::high_resolution_clock::now();

  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, globalWorkSize,
                               localWorkSize, 0, NULL, NULL);
  clFinish(cmd_queue);

  float *device_output;
  device_output = (float *)malloc(N * N * sizeof(float));
  err =
      clEnqueueReadBuffer(cmd_queue, out_parallel_mem, CL_TRUE, 0,
                          N * N * sizeof(float), device_output, 0, NULL, NULL);

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                       time_end - time_start)
                       .count();
  // printArray(device_output);
  std::cout << "\033[0;31m\tExe. Time: " << time_diff << " ms\033[0m"
            << std::endl;

  correctnessCheck(device_output, out_sequential);
  free(device_output);

  err = clFlush(cmd_queue);
  err = clReleaseKernel(kernel);
  err = clReleaseProgram(program);
  err = clReleaseMemObject(in_a_mem);
  err = clReleaseMemObject(in_b_mem);
  err = clReleaseMemObject(out_parallel_mem);
  err = clReleaseCommandQueue(cmd_queue);
  err = clReleaseContext(context);
}

int main(void) {
  in_a = (float *)malloc(N * N * sizeof(float));
  in_b = (float *)malloc(N * N * sizeof(float));
  out_parallel = (float *)malloc(N * N * sizeof(float));
  out_sequential = (float *)malloc(N * N * sizeof(float));

  init(in_a, in_b, out_parallel);
  seqGemm();

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

  free(in_a);
  free(in_b);
  free(out_parallel);
  free(out_sequential);

  return 0;
}

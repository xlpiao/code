#include <iostream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "util.h"

class CL {
 public:
  CL() {
    cl_int err = CL_SUCCESS;

    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    std::cout << "\033[0;31m\nNumber of Platforms: \033[0m" << num_platforms
              << std::endl;

    cl_platform_id* platform_ids = new cl_platform_id[num_platforms];
    err = clGetPlatformIDs(num_platforms, platform_ids, NULL);

    for (int i = 0; i < num_platforms; i++) {
      std::cout << "\033[0;31m[Platform #" << i << "]\033[0m" << std::endl;
      util::printPlatformInfo(platform_ids[i]);

      cl_uint num_devices;
      err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                           &num_devices);
      std::cout << "\033[0;31m\n\tNumber of Devices: \033[0m" << num_devices
                << std::endl;

      cl_device_id* device_ids = new cl_device_id[num_devices];
      err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, num_devices,
                           device_ids, &num_devices);
      for (int j = 0; j < num_devices; j++) {
        std::cout << "\033[0;31m\t[Platform #" << i << "][Device #" << j
                  << "]\033[0m" << std::endl;
        util::printDeviceInfo(device_ids[j]);
      }
      delete[] device_ids;
    }
    delete[] platform_ids;
  }

  ~CL() {}
};

int main(void) {
  CL cl;
  return 0;
}

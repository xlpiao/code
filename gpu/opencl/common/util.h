#include <iostream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CHECK(err)                                                         \
  if (err != CL_SUCCESS) {                                                 \
    std::cout << __FILE__ << ":" << __LINE__ << " " << util::toString(err) \
              << std::endl;                                                \
    exit(EXIT_FAILURE);                                                    \
  }

namespace util {
std::string toString(cl_int err) {
  switch (err) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
#ifdef CL_VERSION_1_1
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
#ifdef CL_VERSION_1_2
    case CL_COMPILE_PROGRAM_FAILURE:
      return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
      return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
      return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
      return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifdef CL_VERSION_1_1
    case CL_INVALID_PROPERTY:
      return "CL_INVALID_PROPERTY";
#endif
#ifdef CL_VERSION_1_2
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
      return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
      return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
#ifdef CL_VERSION_2_0
    case CL_INVALID_PIPE_SIZE:
      return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE:
      return "CL_INVALID_DEVICE_QUEUE";
#endif
#ifdef CL_VERSION_2_2
    case CL_INVALID_SPEC_ID:
      return "CL_INVALID_SPEC_ID";
    case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
      return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif
    case -1001:
      return "No platforms found. (Did you put ICD files in /etc/OpenCL?)";
    default:
      return "Unknown Error";
  }
}

void printPlatformInfo(cl_platform_id platform_id) {
  char platform_string[1024];
  cl_int err = CL_SUCCESS;

  // CL_PLATFORM_NAME
  err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME,
                          sizeof(platform_string), &platform_string, NULL);
  std::cout << "\tCL_PLATFORM_NAME:\t\t" << platform_string << std::endl;

  // CL_PLATFORM_VENDOR
  err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR,
                          sizeof(platform_string), &platform_string, NULL);
  std::cout << "\tCL_PLATFORM_VENDOR:\t\t" << platform_string << std::endl;

  // CL_PLATFORM_VERSION
  err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION,
                          sizeof(platform_string), &platform_string, NULL);
  std::cout << "\tCL_PLATFORM_VERSION:\t\t" << platform_string << std::endl;

  // CL_PLATFORM_PROFILE
  err = clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE,
                          sizeof(platform_string), &platform_string, NULL);
  std::cout << "\tCL_PLATFORM_PROFILE:\t\t" << platform_string << std::endl;

  // CL_PLATFORM_EXTENSIONS
  err = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS,
                          sizeof(platform_string), &platform_string, NULL);
  std::cout << "\tCL_PLATFORM_EXTENSIONS:\t\t" << platform_string << std::endl;
}

void printDeviceInfo(cl_device_id device_id) {
  char device_string[1024];
  cl_int err = CL_SUCCESS;

  // CL_DEVICE_NAME
  err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string),
                        &device_string, NULL);
  std::cout << "\t\tCL_DEVICE_NAME: \t\t\t" << device_string << std::endl;

  // CL_DEVICE_VENDOR
  err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(device_string),
                        &device_string, NULL);
  std::cout << "\t\tCL_DEVICE_VENDOR: \t\t\t" << device_string << std::endl;

  // CL_DRIVER_VERSION
  err = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(device_string),
                        &device_string, NULL);
  std::cout << "\t\tCL_DRIVER_VERSION: \t\t\t" << device_string << std::endl;

  // CL_DEVICE_INFO
  cl_device_type type;
  err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  if (type & CL_DEVICE_TYPE_CPU)
    std::cout << "\t\tCL_DEVICE_TYPE:\t\t\t\tCL_DEVICE_TYPE_CPU" << std::endl;
  if (type & CL_DEVICE_TYPE_GPU)
    std::cout << "\t\tCL_DEVICE_TYPE:\t\t\t\tCL_DEVICE_TYPE_GPU" << std::endl;
  if (type & CL_DEVICE_TYPE_ACCELERATOR)
    std::cout << "\t\tCL_DEVICE_TYPE:\t\t\tCL_DEVICE_TYPE_ACCELERATOR"
              << std::endl;
  if (type & CL_DEVICE_TYPE_DEFAULT)
    std::cout << "\t\tCL_DEVICE_TYPE:\t\t\tCL_DEVICE_TYPE_DEFAULT" << std::endl;

  // CL_DEVICE_MAX_COMPUTE_UNITS
  cl_uint compute_units;
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(compute_units), &compute_units, NULL);
  std::cout << "\t\tCL_DEVICE_MAX_COMPUTE_UNITS:\t\t" << compute_units
            << std::endl;

  // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  size_t workitem_dims;
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                        sizeof(workitem_dims), &workitem_dims, NULL);
  std::cout << "\t\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t" << workitem_dims
            << std::endl;

  // CL_DEVICE_MAX_WORK_ITEM_SIZES
  size_t workitem_size[3];
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                        sizeof(workitem_size), &workitem_size, NULL);
  std::cout << "\t\tCL_DEVICE_MAX_WORK_ITEM_SIZES:\t\t" << workitem_size[0]
            << " / " << workitem_size[1] << " / " << workitem_size[2]
            << std::endl;

  // CL_DEVICE_MAX_WORK_GROUP_SIZE
  size_t workgroup_size;
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        sizeof(workgroup_size), &workgroup_size, NULL);
  std::cout << "\t\tCL_DEVICE_MAX_WORK_GROUP_SIZE:\t\t" << workgroup_size
            << std::endl;

  // CL_DEVICE_MAX_CLOCK_FREQUENCY
  cl_uint clock_frequency;
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                        sizeof(clock_frequency), &clock_frequency, NULL);
  std::cout << "\t\tCL_DEVICE_MAX_CLOCK_FREQUENCY:\t\t" << clock_frequency
            << " MHz" << std::endl;

  // CL_DEVICE_ADDRESS_BITS
  cl_uint addr_bits;
  err = clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits),
                        &addr_bits, NULL);
  std::cout << "\t\tCL_DEVICE_ADDRESS_BITS:\t\t\t" << addr_bits << std::endl;

  // CL_DEVICE_MAX_MEM_ALLOC_SIZE
  cl_ulong max_mem_alloc_size;
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                        sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
  std::cout << "\t\tCL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t"
            << (unsigned int)(max_mem_alloc_size / (1024 * 1024)) << " MByte"
            << std::endl;

  // CL_DEVICE_GLOBAL_MEM_SIZE
  cl_ulong mem_size;
  err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size),
                        &mem_size, NULL);
  std::cout << "\t\tCL_DEVICE_GLOBAL_MEM_SIZE:\t\t"
            << (unsigned int)(mem_size / (1024 * 1024)) << " MByte"
            << std::endl;

  // CL_DEVICE_ERROR_CORRECTION_SUPPORT
  cl_bool error_correction_support;
  err = clGetDeviceInfo(device_id, CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                        sizeof(error_correction_support),
                        &error_correction_support, NULL);
  std::cout << "\t\tCL_DEVICE_ERROR_CORRECTION_SUPPORT:\t"
            << (error_correction_support == CL_TRUE ? "yes" : "no")
            << std::endl;

  // CL_DEVICE_LOCAL_MEM_TYPE
  cl_device_local_mem_type local_mem_type;
  err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE,
                        sizeof(local_mem_type), &local_mem_type, NULL);
  std::cout << "\t\tCL_DEVICE_LOCAL_MEM_TYPE:\t\t"
            << (local_mem_type == 1 ? "local" : "global") << std::endl;

  // CL_DEVICE_LOCAL_MEM_SIZE
  err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size),
                        &mem_size, NULL);
  std::cout << "\t\tCL_DEVICE_LOCAL_MEM_SIZE:\t\t"
            << (unsigned int)(mem_size / 1024) << " KByte" << std::endl;

  // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                        sizeof(mem_size), &mem_size, NULL);
  std::cout << "\t\tCL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t"
            << (unsigned int)(mem_size / 1024) << " KByte" << std::endl;

  // CL_DEVICE_QUEUE_PROPERTIES
  cl_command_queue_properties queue_properties;
  err = clGetDeviceInfo(device_id, CL_DEVICE_QUEUE_PROPERTIES,
                        sizeof(queue_properties), &queue_properties, NULL);
  if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    std::cout << "\t\tCL_DEVICE_QUEUE_PROPERTIES:\t\t"
              << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE" << std::endl;
  if (queue_properties & CL_QUEUE_PROFILING_ENABLE)
    std::cout << "\t\tCL_DEVICE_QUEUE_PROPERTIES:\t\t"
              << "CL_QUEUE_PROFILING_ENABLE" << std::endl;

  // CL_DEVICE_IMAGE_SUPPORT
  cl_bool image_support;
  err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT,
                        sizeof(image_support), &image_support, NULL);
  std::cout << "\t\tCL_DEVICE_IMAGE_SUPPORT:\t\t" << image_support << std::endl;

  // CL_DEVICE_MAX_READ_IMAGE_ARGS
  cl_uint max_read_image_args;
  err =
      clGetDeviceInfo(device_id, CL_DEVICE_MAX_READ_IMAGE_ARGS,
                      sizeof(max_read_image_args), &max_read_image_args, NULL);
  std::cout << "\t\tCL_DEVICE_MAX_READ_IMAGE_ARGS:\t\t" << max_read_image_args
            << std::endl;

  // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
  cl_uint max_write_image_args;
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
                        sizeof(max_write_image_args), &max_write_image_args,
                        NULL);
  std::cout << "\t\tCL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t\t" << max_write_image_args
            << std::endl;

  // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
  cl_uint address_bits_args;
  err = clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS,
                        sizeof(address_bits_args), &address_bits_args, NULL);
  std::cout << "\t\tCL_DEVICE_ADRESS_BITS:\t\t\t" << address_bits_args
            << std::endl;

  // CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
  // CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT,
  // CL_DEVICE_IMAGE3D_MAX_DEPTH
  size_t szMaxDims[5];
  err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t),
                        &szMaxDims[0], NULL);
  err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t),
                        &szMaxDims[1], NULL);
  err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t),
                        &szMaxDims[2], NULL);
  err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t),
                        &szMaxDims[3], NULL);
  err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t),
                        &szMaxDims[4], NULL);
  std::cout << "\t\tCL_DEVICE_IMAGE <dim>:"
            << "\t\t\t2D_MAX_WIDTH\t" << szMaxDims[0]
            << "\n\t\t\t\t\t\t\t2D_MAX_HEIGHT\t" << szMaxDims[1]
            << "\n\t\t\t\t\t\t\t3D_MAX_WIDTH\t" << szMaxDims[2]
            << "\n\t\t\t\t\t\t\t3D_MAX_HEIGHT\t" << szMaxDims[3]
            << "\n\t\t\t\t\t\t\t3D_MAX_DEPTH\t" << szMaxDims[4] << std::endl;

  cl_uint vec_width[6];
  err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                        sizeof(cl_uint), &vec_width[0], NULL);
  err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                        sizeof(cl_uint), &vec_width[1], NULL);
  err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                        sizeof(cl_uint), &vec_width[2], NULL);
  err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                        sizeof(cl_uint), &vec_width[3], NULL);
  err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                        sizeof(cl_uint), &vec_width[4], NULL);
  err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                        sizeof(cl_uint), &vec_width[5], NULL);
  std::cout << "\t\tCL_DEVICE_PREFERRED_VECTOR_WIDTH <type>:"
            << "CHAR " << vec_width[0] << "\n\t\t\t\t\t\t\tSHORT "
            << vec_width[1] << "\n\t\t\t\t\t\t\tINT " << vec_width[2]
            << "\n\t\t\t\t\t\t\tFLOAT " << vec_width[3]
            << "\n\t\t\t\t\t\t\tDOUBLE " << vec_width[4] << "\n"
            << std::endl;
}
}  // namespace util

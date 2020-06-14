project(OPENCL)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G") ## '-G' means do not optimize inst.

## header/lib path
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY})
message(STATUS "INFO ${CUDA_INCLUDE_DIRS}")
message(STATUS "INFO ${CUDA_LIBRARY}")

## library
if(APPLE)
  message(STATUS "INFO APPLE")
  set(LIBS "-framework OpenCL")
else()
  message(STATUS "INFO LINUX")
  set(LIBS "OpenCL -lpthread")
endif()

## build options
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif(NOT CMAKE_BUILD_TYPE)

## internal header
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

## binary
add_executable(bin.atomic_add ${CMAKE_CURRENT_SOURCE_DIR}/src/test/atomic_add.cu)
# target_link_libraries(bin.atomic_add)

add_executable(bin.08_index ${CMAKE_CURRENT_SOURCE_DIR}/src/masterclass/08_index.cu)
add_executable(bin.dim ${CMAKE_CURRENT_SOURCE_DIR}/src/masterclass/dim.cu)
add_executable(bin.dim_v1 ${CMAKE_CURRENT_SOURCE_DIR}/src/masterclass/dim_v1.cu)
add_executable(bin.warp_divergence ${CMAKE_CURRENT_SOURCE_DIR}/src/masterclass/warp_divergence.cu)
add_executable(bin.reduction ${CMAKE_CURRENT_SOURCE_DIR}/src/masterclass/reduction.cu)

project(OPENCL)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

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
add_executable(bin.atomic_add ${CMAKE_CURRENT_SOURCE_DIR}/src/atomic_add.cu)
target_link_libraries(bin.atomic_add)

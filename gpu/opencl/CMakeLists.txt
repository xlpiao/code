project(OPENCL)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## header/lib path
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
message(STATUS "INFO ${OpenCL_INCLUDE_DIRS}")
message(STATUS "INFO ${OpenCL_LIBRARY}")

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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # util.h

## binary
add_executable(bin.QueryInfo ${CMAKE_CURRENT_SOURCE_DIR}/QueryInfo/main.cc)
target_link_libraries(bin.QueryInfo ${LIBS})

add_executable(bin.Gemm ${CMAKE_CURRENT_SOURCE_DIR}/Gemm/main.cc)
target_link_libraries(bin.Gemm ${LIBS})

add_executable(bin.MultiQueue ${CMAKE_CURRENT_SOURCE_DIR}/MultiQueue/main.cc)
target_link_libraries(bin.MultiQueue ${LIBS})

add_executable(bin.Saxpy ${CMAKE_CURRENT_SOURCE_DIR}/Saxpy/main.cc)
target_link_libraries(bin.Saxpy ${LIBS})

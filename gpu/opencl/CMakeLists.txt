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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

## binary
add_executable(bin.query_info ${CMAKE_CURRENT_SOURCE_DIR}/src/query_info/main.cc)
target_link_libraries(bin.query_info ${LIBS})

add_executable(bin.gemm ${CMAKE_CURRENT_SOURCE_DIR}/src/gemm/main.cc)
target_link_libraries(bin.gemm ${LIBS})

add_executable(bin.multi_queue ${CMAKE_CURRENT_SOURCE_DIR}/src/multi_queue/main.cc)
target_link_libraries(bin.multi_queue ${LIBS})

add_executable(bin.saxpy ${CMAKE_CURRENT_SOURCE_DIR}/src/saxpy/main.cc)
target_link_libraries(bin.saxpy ${LIBS})

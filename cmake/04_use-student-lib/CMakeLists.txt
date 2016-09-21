cmake_minimum_required(VERSION 2.8.9)
project (TestLibrary)

## For the shared library:
set ( PROJECT_LINK_LIBS libtestStudent.so )
link_directories( ~/workspace/code/cmake/02_student-lib-shared/build )

## For the static library:
# set ( PROJECT_LINK_LIBS libtestStudent.a )
# link_directories( ~/workspace/code/cmake/03_student-lib-static/build )

include_directories(~/workspace/code/cmake/02_student-lib-shared/include)

add_executable(libtest libtest.cpp)
target_link_libraries(libtest ${PROJECT_LINK_LIBS} )

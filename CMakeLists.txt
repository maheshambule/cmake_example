cmake_minimum_required(VERSION 3.10)
project(cmake_example)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow/lib)

set(CMAKE_CXX_STANDARD 17)
# add_executable(example main.cpp ../../src/Model.cpp ../../src/Tensor.cpp)





add_subdirectory(pybind11)
pybind11_add_module(cmake_example src/example.cpp src/Model.cpp src/Tensor.cpp)

target_include_directories(cmake_example PRIVATE ../../include $ENV{HOME}/libtensorflow/include)

target_link_libraries (cmake_example PRIVATE "${TENSORFLOW_LIB}")
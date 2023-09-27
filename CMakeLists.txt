cmake_minimum_required(VERSION 3.27.3 FATAL_ERROR)
project(cuda-demo VERSION 0.1.0 LANGUAGES CXX CUDA) # ${PROJECT_NAME}

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)

add_executable(${PROJECT_NAME} main.cu)
add_executable(${PROJECT_NAME}-cpu main.cpp)

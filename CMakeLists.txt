cmake_minimum_required(VERSION 3.27.3 FATAL_ERROR)
project(cuda-demo VERSION 0.1.0 LANGUAGES CXX CUDA) # ${PROJECT_NAME}

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	# Enable asan
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
	# Enable msan
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")
	# Enable tsan
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
endif()

# If nvcc is available compile cuda demo
add_executable(${PROJECT_NAME} main.cu)
add_executable(${PROJECT_NAME}-cpu main.cpp)

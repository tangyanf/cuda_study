cmake_minimum_required(VERSION 3.14)
project(nvidia_sgemm)

# set(CUDA_NVCC_FLAGS -g;-G)
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")
set(CUDA_NVCC_FLAGS -arch=compute_70;-code=compute_70)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "------------${SRC}")
CUDA_ADD_EXECUTABLE(sgemm matrix_mult.cu ${SRC})
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

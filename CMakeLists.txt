cmake_minimum_required(VERSION 3.10)
project(cuda_pracitce)

find_package(CUDA REQUIRED)
message(STATUS "CUA_LIBRARY:" ${CUDA_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV LIBRARIES:" ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

cuda_add_executable(add
        add.cu)
cuda_add_executable(cudaDevice cudaDevice.cu)
target_link_libraries(add ${CUDA_LIBRARIES})
target_link_libraries(cudaDevice ${CUDA_LIBRARIES})

cuda_add_executable(Sobel Sobel.cu)
target_link_libraries(Sobel ${CUDA_LIBRARIES}
        ${OpenCV_LIBS})
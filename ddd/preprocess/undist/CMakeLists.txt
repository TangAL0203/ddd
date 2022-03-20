cmake_minimum_required (VERSION 3.15)

project(undistort)

# Requires OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("-- Found OpenCV")
endif(OpenCV_FOUND)

find_package(CUDA REQUIRED)

#add_library(undistort_cuda STATIC ${_CUDA_SRCS})

#include_directories(${CUDA_INCLUDE_DIRS})

#target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY} ${engine_lib})
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${engine_lib} gflags
#        ${PROTOBUF_LIBRARIES} ${CUDA_LIBRARIES})


cuda_add_executable(undistort src/main.cpp src/undistort.cpp src/undistort_cuda.cu src/remap.cu)
#add_executable(undistort src/main.cpp src/undistort.cpp src/undistort_cuda.cu)
#add_executable(undistort src/main.cpp src/undistort.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
include_directories(${CUDA_INCLUDE_DIRS})

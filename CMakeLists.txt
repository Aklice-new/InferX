cmake_minimum_required(VERSION 3.16)
project(InferX LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_BUILD_TYPE "Debug")

add_definitions(-DENABLE_CUDA)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS " -g -O0 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS " -O3 -g ${CMAKE_CXX_FLAGS}")
endif()

option(ENABLE_CUDA "Enable CUDA" ON)


#########################
## Find Dependencies
#########################

# opencv 4.6.0
set(OpenCV_DIR "/home/aklice/SoftWare/ThirdParty/OpenCV/opencv-4.6.0/build")
find_package(OpenCV 4.6.0 REQUIRED)


include(cmake/cuda.cmake)
include(cmake/FindGlog.cmake)

# cutlass
# include_directories(${PROJECT_SOURCE_DIR}/third_party/cutlass-3.5.0/include)

# spdlog
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog/include)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SRC_FILES 
        src/core/*.cpp
        src/graph/*.cpp
        src/graph/pnnx/*.cpp
        src/layer/*.cpp
        src/layer/kernels/*.cpp 
        src/layer/kernels/cpu/*.cpp 
        src/layer/kernels/cuda/*.cu
        src/parser/*.cpp)

add_library(InferX SHARED ${SRC_FILES})
if(ENABLE_CUDA)
    target_link_libraries(InferX CUDA::cudart CUDA::cublas ${GLOG_LIBRARIES} )
else()
    target_link_libraries(InferX ${GLOG_LIBRARIES} )
endif()

#####################################################
### test 
#####################################################

macro(add_test name)
    add_executable(${name} test/${name}/${name}.cpp)
    target_link_libraries(${name} InferX ${OpenCV_LIBS})
endmacro()

add_test(test_tensor)
add_test(test_graph)
add_test(test_parser)
add_test(test_mobilenet)


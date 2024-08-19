cmake_minimum_required(VERSION 3.16)
project(InferX LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")


#########################
## Find Dependencies
#########################


include(cmake/cuda.cmake)
include(cmake/FindGlog.cmake)

# cutlass
# include_directories(${PROJECT_SOURCE_DIR}/third_party/cutlass-3.5.0/include)

# spdlog
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog/include)


file(GLOB SRC_FILES 
        src/core/*.cpp
        src/graph/*.cpp
        src/graph/pnnx/*.cpp
        src/layer/*.cpp
        src/layer/kernels/*.cpp 
        src/layer/kernels/cpu/*.cpp 
        src/layer/kernels/cuda/*.cu)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(InferX SHARED ${SRC_FILES})
target_link_libraries(InferX CUDA::cudart CUDA::cublas ${GLOG_LIBRARIES})


#####################################################
### test 
#####################################################

macro(add_test name)
    add_executable(${name} test/${name}.cpp)
    target_link_libraries(${name} InferX)
endmacro()

add_test(test_tensor)
add_test(test_graph)


cmake_minimum_required(VERSION 3.16)
project(InferX LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

#########################
## Debug settings
#########################
set(CMAKE_BUILD_TYPE "Debug")

add_definitions(-DENABLE_CUDA)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS " -g -O0 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS " -O3 -g ${CMAKE_CXX_FLAGS}")
endif()

option(ENABLE_CUDA "Enable CUDA" ON)
option(ENABLE_OpenMP "Enable OpenMP" ON)
option(ENABLE_examples "Enable examples" ON)

#########################
## Find Dependencies
#########################

include(cmake/cuda.cmake)
include(cmake/FindGlog.cmake)

# cutlass
# include_directories(${PROJECT_SOURCE_DIR}/third_party/cutlass-3.5.0/include)

# spdlog
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog/include)

if(ENABLE_CUDA)
    set(LIBS ${LIBS} CUDA::cudart CUDA::cublas)
endif()

if(ENABLE_OpenMP)
    find_package(OpenMP REQUIRED)
    set(LIBS ${LIBS} OpenMP::OpenMP_CXX)
endif()

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
target_link_libraries(InferX ${LIBS})


if(ENABLE_examples)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()


#####################################################
### test 
#####################################################

macro(add_test name)
    add_executable(${name} test/${name}/${name}.cpp)
    target_link_libraries(${name} InferX ${OpenCV_LIBS})
endmacro()

if(ENABLE_examples)
    add_test(test_tensor)
    add_test(test_graph)
    add_test(test_parser)
    add_test(test_mobilenet)
    add_test(test_resnet)
endif()


cmake_minimum_required(VERSION 3.16)
project(InferX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#########################
## Debug settings
#########################
set(CMAKE_BUILD_TYPE "Debug")

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

include(cmake/FindGlog.cmake)
find_package(GTest REQUIRED)
SET(LIBS ${LIBS} ${GLOG_LIBRARIES})

if(ENABLE_CUDA)
    include(cmake/cuda.cmake)
    message(STATUS "CUDA enabled")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
    enable_language(CUDA)
    set(CUDA_LIBS ${CUDA_LIBS} CUDA::cudart CUDA::cublas)
    add_definitions(-DENABLE_CUDA)
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
        src/parser/*.cpp
        src/utils/*.cpp)

add_library(InferX SHARED ${SRC_FILES})
target_link_libraries(InferX ${LIBS})

if(ENABLE_CUDA)
    file(GLOB CUDA_SRC_FILES src/layer/kernels/cuda/*.cu)
    add_library(InferX_CUDA SHARED ${CUDA_SRC_FILES})
    target_link_libraries(InferX InferX_CUDA ${CUDA_LIBS})
endif()

#####################################################
### test 
#####################################################

macro(add_test name)
    add_executable(${name} test/${name}/${name}.cpp)
    target_link_libraries(${name} InferX ${OpenCV_LIBS} ${GLOG_LIBRARIES} GTest::GTest GTest::Main)
endmacro()

if(ENABLE_examples)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_test(test_tensor)
    add_test(test_graph)
    add_test(test_parser)
    add_test(test_mobilenet_v2)
    add_test(test_resnet)
    add_test(test_maxpooling)
    add_test(test_convolution2d)
    add_test(test_adaptiveAvgPooling)
    add_test(test_sigmoid)
    add_test(test_relu)
    add_test(test_im2col)
    add_test(test_linear)
    add_test(test_hardsigmoid)
    add_test(test_hardswish)
endif()


cmake_minimum_required(VERSION 3.21)
project(SuperPoint)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} \
-isystem /usr/include/c++/11 \
-isystem /usr/include/x86_64-linux-gnu/c++/11 \
-isystem /usr/include/x86_64-linux-gnu/c++/11/bits \
")

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(superpoint_onnx STATIC src/superpoint.cpp)
target_include_directories(
  superpoint_onnx PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include> ${Eigen3_INCLUDE_DIRS})

target_link_libraries(superpoint_onnx PUBLIC ${OpenCV_LIBS}
                                             onnxruntime::onnxruntime)

add_executable(superpoint_onnx_demo src/demo.cpp)
target_link_libraries(
  superpoint_onnx_demo PRIVATE superpoint_onnx ${OpenCV_LIBS}
                               onnxruntime::onnxruntime)

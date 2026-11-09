cmake_minimum_required(VERSION 3.16)
project(classify CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(classify src/main.cpp)
target_link_libraries(classify openvino::runtime ${OpenCV_LIBS})
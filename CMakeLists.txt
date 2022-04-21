cmake_minimum_required(VERSION 3.21)
project(ceres-translate-example)

set(CMAKE_CXX_STANDARD 20)

add_executable(ceres-translate-example main.cpp)

find_package(Ceres REQUIRED)
target_link_libraries(ceres-translate-example Ceres::ceres)

#find_package(OpenCV COMPONENTS core highgui)
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#target_link_libraries(ceres-translate-example ${OpenCV_LIBRARIES})

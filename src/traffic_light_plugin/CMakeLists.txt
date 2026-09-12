cmake_minimum_required(VERSION 3.0.2)
project(traffic_light_plugin)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED)
find_package(gazebo REQUIRED)

catkin_package()

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(traffic_light_plugin
  src/traffic_light_plugin.cpp
)

target_link_libraries(traffic_light_plugin
  ${GAZEBO_LIBRARIES}
)

install(TARGETS traffic_light_plugin
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
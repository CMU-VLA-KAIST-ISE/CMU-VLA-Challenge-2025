cmake_minimum_required(VERSION 2.8.3)
project(interaction_manager)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs
)
catkin_install_python(PROGRAMS
  src/interaction_manager.py
  src/launch_pipeline.py
  src/path.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
) 
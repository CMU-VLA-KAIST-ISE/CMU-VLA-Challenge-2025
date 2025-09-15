cmake_minimum_required(VERSION 3.0.2)
project(leo_vlm)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  src/leo_llm.py
  src/planner_partialSG.py
  src/gpt_llm.py
  src/gpt_llm_4o.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
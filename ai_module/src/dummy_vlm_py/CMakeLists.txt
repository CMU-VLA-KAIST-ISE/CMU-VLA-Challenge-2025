cmake_minimum_required(VERSION 3.0.2)
project(dummy_vlm_py)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/dummyVLM.py
  scripts/graph_builder_node.py
  scripts/llm_prompt_builder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
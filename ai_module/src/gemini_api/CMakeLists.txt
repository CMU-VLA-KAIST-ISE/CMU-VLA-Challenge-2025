cmake_minimum_required(VERSION 3.0.2)
project(gemini_api)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  tf
  tf2_ros
)

catkin_package()

# Python 노드 설치
# - fusion_pipeline_node.py 는 scripts/ 에 있다고 가정
# - object_selector_gemini.py 는 기존 위치(src/) 유지해도 됨
catkin_install_python(PROGRAMS
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# launch 파일 설치
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

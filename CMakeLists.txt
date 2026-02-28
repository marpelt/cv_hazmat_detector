cmake_minimum_required(VERSION 3.0.2)
project(cv_hazmat_detector)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  cv_msg
)

catkin_package(
  CATKIN_DEPENDS rospy cv_msg
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

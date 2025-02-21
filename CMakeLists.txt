cmake_minimum_required(VERSION 3.16)
project(react_ros)

# Use the FindBoost provided by CMake, rather than the one provided by Boost
# (for CMake >=3.30).
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.30")
  cmake_policy(SET CMP0167 OLD)
endif()

find_package(catkin REQUIRED COMPONENTS hydra_seg_ros react roscpp rospy
                                        sensor_msgs std_msgs)

include_directories(${catkin_INCLUDE_DIRS})

catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS app/react_ros_node.py DESTINATION
                      ${CATKIN_PACKAGE_BIN_DESTINATION})

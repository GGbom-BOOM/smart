cmake_minimum_required(VERSION 3.0.2)
project(person_reid)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  std_srvs
)

catkin_package()

## 装到 lib/person_reid/ 下, 这样 rosrun person_reid reid_node.py 能找到
catkin_install_python(PROGRAMS
  scripts/reid_node.py
  scripts/reid_engine.py
  scripts/osnet.py
  scripts/build_gallery.py
  scripts/calibrate_threshold.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## 权重、图库、launch 装到 share/person_reid/
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
  FILES_MATCHING PATTERN "*.pth"
)

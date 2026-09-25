cmake_minimum_required(VERSION 3.8)
project(inmoov_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/MemoryQuery.srv"
  "action/Speak.action"
  "msg/SoundDirection.msg"
  "msg/JointCommand.msg"
  DEPENDENCIES std_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

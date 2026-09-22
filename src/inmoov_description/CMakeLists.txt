cmake_minimum_required(VERSION 3.8)
project(inmoov_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY description config launch meshes
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()

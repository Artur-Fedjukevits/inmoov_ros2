cmake_minimum_required(VERSION 3.8)
project(inmoov_description)

find_package(ament_cmake REQUIRED)

# The Blender source (.blend) is for editing the model, not needed at runtime.
install(
  DIRECTORY description config launch meshes
  DESTINATION share/${PROJECT_NAME}/
  PATTERN "*.blend" EXCLUDE
)

# Meshes are CC BY-NC 4.0, the rest GPL-3.0 — ship both notices with them.
install(
  FILES LICENSE LICENSE-ASSETS.md
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()

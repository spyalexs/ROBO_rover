cmake_minimum_required(VERSION 3.8)
project(robo_rover)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  ${PROJECT_NAME}/rover_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME rover_node
)

file(GLOB RESOURCE ${CMAKE_CURRENT_SOURCE_DIR}/resource/*.yaml)
message( ${RESOURCE})


# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)


#Install the resource riles
install(PROGRAMS
  ${RESOURCE}
  DESTINATION share/${PROJECT_NAME}/resource
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
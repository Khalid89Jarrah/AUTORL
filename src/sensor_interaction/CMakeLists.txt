cmake_minimum_required(VERSION 3.8)
project(sensor_interaction)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(ros_gz_sim REQUIRED)
find_package(ros_gz_interfaces REQUIRED)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Include directories
include_directories(include)


# Install Python module
ament_python_install_package(${PROJECT_NAME})

# Find all Python files in the 'scripts' directory
file(GLOB PYTHON_SCRIPTS "scripts/*.py")

# Install the Python scripts
install(PROGRAMS
  ${PYTHON_SCRIPTS}
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY resources
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

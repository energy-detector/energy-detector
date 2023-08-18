cmake_minimum_required(VERSION 3.10)
project(target_msg)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

###########
## Build ##
###########

file(GLOB ${PROJECT_NAME}_msg_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} msg/*)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  ${${PROJECT_NAME}_msg_files}
  DEPENDENCIES
  ${${PROJECT_NAME}_FOUND_BUILD_DEPENDS}
)

#############
## Install ##
#############

ament_auto_package()
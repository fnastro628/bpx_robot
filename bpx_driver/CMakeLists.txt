cmake_minimum_required(VERSION 3.10)
project(bpx_driver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp        REQUIRED)
find_package(sensor_msgs   REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs      REQUIRED)
find_package(std_msgs      REQUIRED)
find_package(std_srvs      REQUIRED)
find_package(tf2           REQUIRED)
find_package(tf2_ros       REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Threads       REQUIRED)

# ── BPX SDK ──────────────────────────────────────────────────────────────────
# Assumes the bpx_sdk_open repo lives beside bpx_robot/
set(BPX_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../bpx_sdk_open")
set(BPX_SDK_INCLUDE "${BPX_SDK_ROOT}/include")

# Select the correct .so for the build architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(BPX_SDK_LIB "${BPX_SDK_ROOT}/lib/libbpx_sdk_aarch64.so")
else()
  set(BPX_SDK_LIB "${BPX_SDK_ROOT}/lib/libbpx_sdk_x86_64.so")
endif()

message(STATUS "BPX SDK lib: ${BPX_SDK_LIB}")

# ── Target ───────────────────────────────────────────────────────────────────
add_executable(bpx_node src/bpx_node.cpp)

target_include_directories(bpx_node PRIVATE
  include
  ${BPX_SDK_INCLUDE}
)

ament_target_dependencies(bpx_node
  rclcpp sensor_msgs geometry_msgs nav_msgs
  std_msgs std_srvs tf2 tf2_ros tf2_geometry_msgs
)

target_link_libraries(bpx_node
  ${BPX_SDK_LIB}
  Threads::Threads
)

# Install node binary
install(TARGETS bpx_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install the BPX shared lib so the node can find it at runtime
install(FILES ${BPX_SDK_LIB}
  DESTINATION lib
)

ament_package()

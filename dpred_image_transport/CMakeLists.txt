cmake_minimum_required(VERSION 3.20)

project(dpred_image_transport)

# The vendored depth codec uses C++20 (std::bit_cast, std::countr_zero).
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(image_transport REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# zstd: use the CMake config package if the distro ships one, otherwise the
# plain library. Prefer the shared target: distro static libzstd.a is not
# built with -fPIC and cannot be linked into this shared plugin library.
find_package(zstd CONFIG QUIET)
if(TARGET zstd::libzstd)
  set(ZSTD_DEPENDENCY zstd::libzstd)
elseif(TARGET zstd::libzstd_shared)
  set(ZSTD_DEPENDENCY zstd::libzstd_shared)
elseif(TARGET zstd::libzstd_static)
  set(ZSTD_DEPENDENCY zstd::libzstd_static)
else()
  find_path(ZSTD_INCLUDE_DIR zstd.h REQUIRED)
  find_library(ZSTD_LIBRARY NAMES zstd REQUIRED)
  set(ZSTD_DEPENDENCY ${ZSTD_LIBRARY})
  include_directories(${ZSTD_INCLUDE_DIR})
endif()

include_directories(include)

add_library(
  ${PROJECT_NAME} SHARED
  src/depth_codec.cpp
  src/dpred_publisher.cpp
  src/dpred_subscriber.cpp
  src/manifest.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${ZSTD_DEPENDENCY}
  image_transport::image_transport
  rclcpp::rclcpp
  pluginlib::pluginlib
  ${sensor_msgs_TARGETS}
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY "include/"
  DESTINATION include
)
pluginlib_export_plugin_description_file(image_transport dpred_plugins.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_depth_codec test/test_depth_codec.cpp)
  target_include_directories(test_depth_codec PRIVATE src)
  target_link_libraries(test_depth_codec ${PROJECT_NAME})
endif()

ament_package()

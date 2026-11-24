cmake_minimum_required(VERSION 3.8)
project(dobot_twin_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch rviz config tools docs
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS scripts/instance_labels.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(model_verification test/test_model_verification.py
    TIMEOUT 120)
endif()

ament_package()

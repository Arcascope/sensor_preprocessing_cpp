cmake_minimum_required(VERSION 3.12)
project(senpy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# add ../cpp relative to this CMakeLists.txt
set(CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/../src")
# add headers in ../cpp to the target include path
# (will apply to the target once it's created)
include_directories(${CPP_DIR})

# Find pybind11
find_package(pybind11 REQUIRED)

# Create the Python module
pybind11_add_module(senpy 
    sensor_processing_native.cpp
)

# expose include directory to the target
target_include_directories(senpy PRIVATE ${CPP_DIR})

# Link against necessary libraries
target_link_libraries(senpy PRIVATE)

# Set RPATH for the module
set_target_properties(senpy PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Installation
install(TARGETS senpy LIBRARY DESTINATION .)
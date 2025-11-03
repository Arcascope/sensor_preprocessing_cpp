cmake_minimum_required(VERSION 3.10)
project(sensor_processing_native)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Release for optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2")
endif()

# Find required packages
find_package(PkgConfig)

# Add src/ to include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add the shared library
add_library(sensor_processing SHARED
    sensor_processing_native.cpp
    # sensor_processing.cpp is included via #include
)

# Set library properties
set_target_properties(sensor_processing PROPERTIES
    CXX_VISIBILITY_PRESET default
    VERSION 1.0.0
    SOVERSION 1
)

# Include directories
target_include_directories(sensor_processing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Platform-specific settings
if(ANDROID)
    # Android-specific settings
    find_library(log-lib log)
    target_link_libraries(sensor_processing ${log-lib})
    
elseif(APPLE)
    # macOS/iOS-specific settings
    set_target_properties(sensor_processing PROPERTIES
        MACOSX_RPATH TRUE
        INSTALL_RPATH "@loader_path"
    )
    
elseif(WIN32)
    # Windows-specific settings
    target_compile_definitions(sensor_processing PRIVATE WIN32_LEAN_AND_MEAN)
    
elseif(UNIX)
    # Linux-specific settings
    target_link_libraries(sensor_processing m pthread)
endif()

# Install rules
install(TARGETS sensor_processing
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
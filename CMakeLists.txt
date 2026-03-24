cmake_minimum_required(VERSION 3.24)
project(sensor_processing_native)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Release for optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimizations
# Note: -ffast-math is intentionally omitted — it breaks FINUFFT (IEEE-754 violations)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2")
endif()

# Apple Clang needs explicit help finding Homebrew's libomp.
# Flags must be semicolon-separated CMake lists, not space-separated strings,
# so that each flag is passed as a separate argument to the compiler.
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_C_FLAGS "-Xpreprocessor;-fopenmp;-I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor;-fopenmp;-I${LIBOMP_PREFIX}/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()

# Download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.8/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# Add finufft dependency
CPMAddPackage("gh:flatironinstitute/finufft@2.5.0")

# Find required packages
find_package(PkgConfig)

# Add src/ to include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add the shared library
add_library(sensor_processing SHARED
    src/sensor_processing_native.cpp
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

# Link finufft and enable its include guard in the source
target_link_libraries(sensor_processing PRIVATE finufft)
target_compile_definitions(sensor_processing PRIVATE USE_FINUFFT)

# Platform-specific settings
if(ANDROID)
    # Android-specific settings
    find_library(log-lib log)
    target_link_libraries(sensor_processing PRIVATE ${log-lib})
    
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
    target_link_libraries(sensor_processing PRIVATE m pthread)
endif()

# Install rules
install(TARGETS sensor_processing
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
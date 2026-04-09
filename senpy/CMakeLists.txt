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

# FINUFFT is required for senpy builds.
set(FINUFFT_INCLUDE_DIR "" CACHE PATH "Path to FINUFFT headers")
set(FINUFFT_LIB "" CACHE FILEPATH "Path to the FINUFFT static library")
set(FFTW3_LIB "" CACHE FILEPATH "Path to the FFTW3 static library")
set(FFTW3F_LIB "" CACHE FILEPATH "Path to the FFTW3F static library")

if(NOT EXISTS "${FINUFFT_INCLUDE_DIR}" OR NOT EXISTS "${FINUFFT_LIB}" OR NOT EXISTS "${FFTW3_LIB}" OR NOT EXISTS "${FFTW3F_LIB}")
    message(FATAL_ERROR "FINUFFT and FFTW artifacts are required to build senpy")
endif()

# Create the Python module with the new name
pybind11_add_module(_core
    ${CPP_DIR}/sensor_processing_native.cpp
)

# Change the output name to match what setup.py expects
set_target_properties(_core PROPERTIES OUTPUT_NAME "_core")

# expose include directory to the target
target_include_directories(_core PRIVATE ${CPP_DIR})

# Add required FINUFFT include directory
target_include_directories(_core PRIVATE ${FINUFFT_INCLUDE_DIR})

# Link against required libraries
target_link_libraries(_core PRIVATE ${FINUFFT_LIB} ${FFTW3_LIB} ${FFTW3F_LIB} m gomp)

# Set RPATH for the module
set_target_properties(_core PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Installation - install into the senpy package directory
install(TARGETS _core 
    LIBRARY DESTINATION senpy
    RUNTIME DESTINATION senpy
)
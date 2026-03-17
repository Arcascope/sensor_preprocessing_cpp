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

# Finufft configuration
set(FINUFFT_INCLUDE_DIR "/tmp/finufft/include")
set(FINUFFT_LIB "/tmp/finufft/build/src/libfinufft.a")
set(FFTW3_LIB "/tmp/finufft/build/_deps/fftw3-build/libfftw3.a")
set(FFTW3F_LIB "/tmp/finufft/build/_deps/fftw3f-build/libfftw3f.a")

if(EXISTS "${FINUFFT_INCLUDE_DIR}" AND EXISTS "${FINUFFT_LIB}")
    message(STATUS "Finufft found at ${FINUFFT_LIB}")
    set(USE_FINUFFT ON)
    add_definitions(-DUSE_FINUFFT)
else()
    message(STATUS "Finufft not found, building without NUFFT support")
    set(USE_FINUFFT OFF)
endif()

# Create the Python module with the new name
pybind11_add_module(_core
    ${CPP_DIR}/sensor_processing_native.cpp
)

# Change the output name to match what setup.py expects
set_target_properties(_core PROPERTIES OUTPUT_NAME "_core")

# expose include directory to the target
target_include_directories(_core PRIVATE ${CPP_DIR})

# Add finufft include directory if available
if(USE_FINUFFT)
    target_include_directories(_core PRIVATE ${FINUFFT_INCLUDE_DIR})
endif()

# Link against necessary libraries
if(USE_FINUFFT)
    target_link_libraries(_core PRIVATE ${FINUFFT_LIB} ${FFTW3_LIB} ${FFTW3F_LIB} m gomp)
else()
    target_link_libraries(_core PRIVATE)
endif()

# Set RPATH for the module
set_target_properties(_core PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Installation - install into the senpy package directory
install(TARGETS _core 
    LIBRARY DESTINATION senpy
    RUNTIME DESTINATION senpy
)
cmake_minimum_required(VERSION 3.15)
project(dnaflex_parsers)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add source files
set(SOURCES
    msa_profile_pybind.cc
    msa_profile.cc
    msa_conversion_pybind.cc
    msa_conversion.cc
    cif_dict_pybind.cc
    fasta_iterator_pybind.cc
    fasta_iterator.cc
    module.cc
)

# Create Python module
pybind11_add_module(parsers_cpp ${SOURCES})

# Include directories
target_include_directories(parsers_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${pybind11_INCLUDE_DIRS}
)

# Set output directory
set_target_properties(parsers_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# Enable optimizations
target_compile_options(parsers_cpp PRIVATE 
    -O3
    -ffast-math
    -march=native
)
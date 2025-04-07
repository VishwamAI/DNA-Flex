cmake_minimum_required(VERSION 3.15)
project(dnaflex_parsers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add the pybind11 module
pybind11_add_module(parsers_cpp
    module.cc
    cif_dict_lib.cc
    cif_dict_pybind.cc
    fasta_iterator_lib.cc
    fasta_iterator_pybind.cc
    msa_conversion_lib.cc
    msa_conversion_pybind.cc
)

# Set include directories
target_include_directories(parsers_cpp PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Link against Python libraries
target_link_libraries(parsers_cpp PRIVATE
    ${PYTHON_LIBRARIES}
)
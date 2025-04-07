cmake_minimum_required(VERSION 3.15)
project(dnaflex_msa)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add the pybind11 module
pybind11_add_module(msa_profile 
    msa_profile_pybind.cc
)

# Set include directories
target_include_directories(msa_profile PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Link against Python libraries
target_link_libraries(msa_profile PRIVATE
    ${PYTHON_LIBRARIES}
)
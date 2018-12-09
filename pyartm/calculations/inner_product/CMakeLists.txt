cmake_minimum_required(VERSION 3.5)
project(inner_product)

include_directories(${PROJECT_SOURCE_DIR}/src)

pybind11_add_module(cpp_impl src/inner1d.hpp src/inner1d.cpp)
set_target_properties(cpp_impl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pyartm/calculations/inner_product"
)

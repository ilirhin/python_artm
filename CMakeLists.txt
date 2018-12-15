cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(THIRD_PARTY_DIRS ${CMAKE_SOURCE_DIR}/third_party)

add_compile_options ("-fexceptions" "-g")
# build type specific
if (CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "Test")
    add_compile_options ("-O0")
    add_definitions ("-DCONSOLE_LOGGER -DTEST_MOCK")
else ()
    add_compile_options ("-O3")
    add_definitions ("-DNDEBUG")
endif ()

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

add_subdirectory(${THIRD_PARTY_DIRS}/pybind11)
add_subdirectory(${CMAKE_SOURCE_DIR}/pyartm/calculations/inner_product)

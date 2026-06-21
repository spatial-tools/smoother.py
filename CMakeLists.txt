cmake_minimum_required(VERSION 3.15...3.27)

project(_smoother LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Python
# SKBUILD_SABI_COMPONENT is set by scikit-build-core when wheel.py-api
# requests a stable ABI build (Development.SABIModule)
find_package(Python 3.11 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# nanobind
# nanobind is installed as a Python package, so we find it through its CMake dir
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE nanobind_FIND_RESULT
    ERROR_QUIET
)

if(NOT nanobind_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR
        "nanobind not found!\n"
        "Please install nanobind:\n"
        "  pip install nanobind\n"
        "\n"
        "If nanobind is installed, make sure you're using the same Python\n"
        "interpreter that has nanobind installed."
    )
endif()

find_package(nanobind CONFIG REQUIRED)

# STABLE_ABI: builds abi3 wheels on Python >= 3.12, regular builds otherwise
nanobind_add_module(_smoother STABLE_ABI src/smooth.cpp)

install(TARGETS _smoother LIBRARY DESTINATION smoother)

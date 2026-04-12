cmake_minimum_required(VERSION 3.15...3.26)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(_smoother)

# C++ files
include_directories("${CMAKE_SOURCE_DIR}/src")

file (GLOB SOURCE_FILES "src/*.cpp")
file (GLOB HEADER_FILES "src/carma/*.h" "src/carma/carma/*.h")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} )

# Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Pybind11
# pybind11 is installed as a Python package, so we find it through Python's get_cmake_dir()
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_FIND_RESULT
    ERROR_QUIET
)

if(pybind11_FIND_RESULT EQUAL 0 AND pybind11_DIR)
    message(STATUS "Found pybind11 CMake dir via Python: ${pybind11_DIR}")
    # Use the directory directly to find pybind11
    find_package(pybind11 REQUIRED CONFIG PATHS ${pybind11_DIR} NO_DEFAULT_PATH)
else()
    # Fallback: try standard find_package (in case pybind11 was installed system-wide)
    message(STATUS "pybind11 not found via Python, trying standard find_package...")
    find_package(pybind11 QUIET CONFIG)

    if(NOT pybind11_FOUND)
        message(FATAL_ERROR
            "pybind11 not found!\n"
            "Please install pybind11:\n"
            "  pip install pybind11\n"
            "  or\n"
            "  poetry add pybind11\n"
            "\n"
            "If pybind11 is installed, make sure you're using the same Python\n"
            "interpreter that has pybind11 installed."
        )
    endif()
endif()

pybind11_add_module(_smoother ${SOURCE_FILES})

# Armadillo (header-only mode using vendored headers)
set(ARMADILLO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/vendor/armadillo/include")
target_include_directories(_smoother PRIVATE ${ARMADILLO_INCLUDE_DIR})
target_compile_definitions(_smoother PRIVATE ARMA_DONT_USE_WRAPPER)

# BLAS and LAPACK (platform-native: Accelerate on macOS, OpenBLAS on Linux)
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(_smoother PRIVATE ${ACCELERATE_FRAMEWORK})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    target_link_libraries(_smoother PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

target_compile_definitions(_smoother PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS _smoother LIBRARY DESTINATION smoother)

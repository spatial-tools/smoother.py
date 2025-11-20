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

# Armadillo
# Try to find Armadillo using CMake's find_package first (works if Armadillo was installed with CMake)
find_package(Armadillo QUIET)

if(NOT Armadillo_FOUND)
    # Fallback: Manual detection for systems where Armadillo doesn't provide CMake config
    message(STATUS "Armadillo not found via find_package, trying manual detection...")
    
    # Check for ARMADILLO_ROOT environment variable (user can set this)
    if(DEFINED ENV{ARMADILLO_ROOT})
        set(ARMADILLO_ROOT $ENV{ARMADILLO_ROOT})
        message(STATUS "Using ARMADILLO_ROOT: ${ARMADILLO_ROOT}")
    endif()
    
    # Find Armadillo headers
    find_path(ARMADILLO_INCLUDE_DIR
        NAMES armadillo
        HINTS
            ${ARMADILLO_ROOT}
            $ENV{ARMADILLO_ROOT}
        PATHS
            /usr/include
            /usr/local/include
            /opt/homebrew/include                     # macOS Homebrew (Apple Silicon)
            /opt/local/include                        # macOS MacPorts
            C:/armadillo/include                      # Windows typical
            C:/vcpkg/installed/*/include              # Windows vcpkg
            $ENV{CONDA_PREFIX}/include                # Conda environments
            ${CMAKE_PREFIX_PATH}/include
        PATH_SUFFIXES
            include
            armadillo
    )
    
    # Find Armadillo library
    # Platform-specific library names
    if(WIN32)
        set(ARMADILLO_LIB_NAMES armadillo armadillo.lib)
    else()
        set(ARMADILLO_LIB_NAMES armadillo libarmadillo.so libarmadillo.dylib)
    endif()
    
    find_library(ARMADILLO_LIB
        NAMES ${ARMADILLO_LIB_NAMES}
        HINTS
            ${ARMADILLO_ROOT}
            $ENV{ARMADILLO_ROOT}
        PATHS
            /usr/lib
            /usr/local/lib
            /usr/lib/x86_64-linux-gnu                # Debian/Ubuntu
            /usr/lib64                               # Some Linux distributions
            /opt/homebrew/lib                        # macOS Homebrew (Apple Silicon)
            /opt/local/lib                           # macOS MacPorts
            C:/armadillo/lib                         # Windows typical
            C:/vcpkg/installed/*/lib                 # Windows vcpkg
            $ENV{CONDA_PREFIX}/lib                   # Conda environments
            ${CMAKE_PREFIX_PATH}/lib
        PATH_SUFFIXES
            lib
            lib64
            Release
            Debug
    )
    
    # Check if we found both header and library
    if(ARMADILLO_INCLUDE_DIR AND ARMADILLO_LIB)
        message(STATUS "Found Armadillo: ${ARMADILLO_LIB}")
        message(STATUS "Armadillo include dir: ${ARMADILLO_INCLUDE_DIR}")
        set(ARMADILLO_FOUND TRUE)
        set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR})
        set(ARMADILLO_LIBRARIES ${ARMADILLO_LIB})
    else()
        set(ARMADILLO_FOUND FALSE)
    endif()
endif()

# Verify Armadillo was found
if(NOT ARMADILLO_FOUND AND NOT Armadillo_FOUND)
    message(FATAL_ERROR 
        "Armadillo library not found!\n"
        "Please install Armadillo:\n"
        "  macOS:    brew install armadillo\n"
        "  Linux:    sudo apt-get install libarmadillo-dev  (Debian/Ubuntu)\n"
        "            sudo yum install armadillo-devel      (RHEL/CentOS)\n"
        "  Windows:  vcpkg install armadillo\n"
        "            or download from http://arma.sourceforge.net/\n"
        "\n"
        "You can also set ARMADILLO_ROOT to point to the installation directory."
    )
endif()

# Use the appropriate variables based on detection method
if(Armadillo_FOUND)
    # Found via find_package
    target_include_directories(_smoother PRIVATE ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(_smoother PUBLIC ${ARMADILLO_LIBRARIES})
else()
    # Found via manual detection
    target_include_directories(_smoother PRIVATE ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(_smoother PUBLIC ${ARMADILLO_LIBRARIES})
endif()

target_compile_definitions(_smoother PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS _smoother LIBRARY DESTINATION .)

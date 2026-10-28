cmake_minimum_required(VERSION 3.16)

project(VapourSynth-RIFE-OpenVINO
    VERSION 1.0
    DESCRIPTION "RIFE frame interpolation for VapourSynth on the OpenVINO runtime (CPU)"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# OpenVINO runtime. Point CMake at an extracted OpenVINO archive with
#   -DOpenVINO_DIR=<archive>/runtime/cmake   or   -DCMAKE_PREFIX_PATH=<archive>
# or install the OpenVINO runtime package of your distribution (apt/brew/conda).
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(Threads REQUIRED)

# VapourSynth headers are vendored in RIFE/ so no VapourSynth SDK is needed to build.
# VS_INSTALL_DIR can be set to the VapourSynth plugin directory to `make install` there.
set(VS_INSTALL_DIR "" CACHE PATH "VapourSynth plugin directory (default: <libdir>/vapoursynth)")

add_library(rife_openvino MODULE
    RIFE/plugin.cpp
    RIFE/rife_ov.cpp
    RIFE/rife_ov.h
    RIFE/VapourSynth4.h
    RIFE/VSHelper4.h)

target_include_directories(rife_openvino PRIVATE RIFE)
target_link_libraries(rife_openvino PRIVATE openvino::runtime Threads::Threads)

if(WIN32)
    # Output: RIFE-OpenVINO.dll
    set_target_properties(rife_openvino PROPERTIES OUTPUT_NAME "RIFE-OpenVINO" PREFIX "")
    target_compile_definitions(rife_openvino PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    if(MSVC)
        target_compile_options(rife_openvino PRIVATE /W3 /utf-8 /EHsc)
    endif()
else()
    # Output: librife_openvino.so (Linux) / librife_openvino.so (macOS as a MODULE)
    set_target_properties(rife_openvino PROPERTIES OUTPUT_NAME "rife_openvino" PREFIX "lib" SUFFIX ".so")
    # Let the plugin find libopenvino.so next to itself (./ or ./openvino/) and in the
    # OpenVINO directory it was linked against, without LD_LIBRARY_PATH tweaks.
    set_target_properties(rife_openvino PROPERTIES
        BUILD_RPATH_USE_ORIGIN ON
        INSTALL_RPATH "$ORIGIN;$ORIGIN/openvino"
        INSTALL_RPATH_USE_LINK_PATH ON)
    target_compile_options(rife_openvino PRIVATE -Wall -Wextra -fno-math-errno -fno-trapping-math)
endif()

# Installation into the VapourSynth plugin directory.
include(GNUInstallDirs)
if(VS_INSTALL_DIR)
    set(_vs_plugin_dir "${VS_INSTALL_DIR}")
else()
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_get_variable(_vs_libdir vapoursynth libdir)
    endif()
    if(_vs_libdir)
        set(_vs_plugin_dir "${_vs_libdir}/vapoursynth")
    else()
        set(_vs_plugin_dir "${CMAKE_INSTALL_FULL_LIBDIR}/vapoursynth")
    endif()
endif()
message(STATUS "VapourSynth plugin install dir: ${_vs_plugin_dir}")
install(TARGETS rife_openvino LIBRARY DESTINATION "${_vs_plugin_dir}")

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

# libopenvino.so pulls in libtbb.so.12 but carries no rpath of its own, so on Linux a
# plugin shipped with the OpenVINO runtime next to it would still need a *system*
# libtbb12. Linking TBB directly (and forcing the DT_NEEDED entry even though the plugin
# itself calls no TBB symbol) makes libtbb resolve through the plugin's $ORIGIN rpath.
#
# The shared OpenVINO package does not export a TBB target itself, so look for the
# oneTBB bundled with the archive (runtime/3rdparty/tbb) first, then a system oneTBB.
if(NOT WIN32)
    if(NOT TARGET TBB::tbb)
        find_package(TBB CONFIG QUIET COMPONENTS tbb
            PATHS "${OpenVINO_DIR}/../3rdparty/tbb" "${OpenVINO_DIR}/../3rdparty/tbb/lib/cmake/TBB"
            NO_DEFAULT_PATH)
    endif()
    if(NOT TARGET TBB::tbb)
        find_package(TBB CONFIG QUIET COMPONENTS tbb)
    endif()
    if(TARGET TBB::tbb)
        message(STATUS "Linking TBB directly: ${TBB_DIR}")
        target_link_libraries(rife_openvino PRIVATE -Wl,--no-as-needed TBB::tbb -Wl,--as-needed)
    else()
        message(WARNING "oneTBB CMake config not found; the plugin will rely on a system libtbb12 at run time")
    endif()
endif()

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
    # The same rpath is used for the build output and the installed file, so a
    # librife_openvino.so copied straight out of build/ (as the CI artifact is)
    # also resolves $ORIGIN first.
    set_target_properties(rife_openvino PROPERTIES
        BUILD_RPATH_USE_ORIGIN ON
        BUILD_WITH_INSTALL_RPATH ON
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

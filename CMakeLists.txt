# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#[[
{xrst_begin cmake_command usr}
{xrst_spell
    dir
    filesystem
    Config
    libtorch
    build build
    gtest
}

Using CMake to Build, Test, and Install ad_tensor
#################################################

Command
*******
{xrst_code sh}
    cmake -S source_dir -B build_dir \
        -G generator \
        -D include_gtest=(true|false) \
        -D include_plugin=(true|false) \
        -D Torch_DIR=torch_dir
{xrst_code}

In addition, you can change the standard cmake settings that begin
with ``CMake_`` ; e.g.,
{xrst_code sh}
    -D CMAKE_INSTALL_PREFIX=install_prefix
{xrst_code}

source_dir
==========
is the top level directory for the ad_tensor source code.

build_dir
=========
is the top level directory where the build files are located.

generator
=========
is which cmake generator to use; e.g. Ninja.

include_gtest
=============
If include_gtest is true (false), the following google targets
will be included (will not be included) in the build:
examples, benchmarks, and tests.
These targets require the google test package.

include_plugin
==============
If include_plugin is true (false), the ability to link
source code generated using :ref:`adfn_src_gen-name` is included in the build.
This will require the system and filesystem components of the boost library.

torch_dir
=========
is the directory containing the TorchConfig.cmake file.
This enables ad_tensor to link to the C++ libtorch package.

Build
*****
The following command will build ad_tensor:
{xrst_code sh}
    cmake --build build_dir
{xrst_code}
where build_dir is the same as above.

Test
****
The following command will run the examples, tests and benchmarks.
These require that include_gtest to be true:
{xrst_code sh}
    build_dir/examples/examples
    build_dir/tests/tests
    build_dir/benchmarks/benchmarks
{xrst_code}
Note that on windows you will have to change '/' to '\' .

Install
*******
The following command will install ad_tensor:
{xrst_code sh}
    cmake --install build_dir
{xrst_code}

{xrst_end cmake_command}
]]
# ---------------------------------------------------------------------------
# print_variable
MACRO(print_variable variable)
    MESSAGE(STATUS "${variable} = ${${variable}}" )
ENDMACRO(print_variable)
#
# cmake_minimum_required
# version 3.12 is required for c++20
cmake_minimum_required(VERSION 3.12)
#
# project
project(ad_tensor VERSION "2026.9.3")
#
# CMAKE_INSTALL_INCLUDEDIR, CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)
print_variable(CMAKE_INSTALL_INCLUDEDIR)
print_variable(CMAKE_INSTALL_LIBDIR)
#
# CXX_STANDARD
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#
# TORCH_INCLUDE_DIRS, TORCH_CXX_FLAGS, TORCH_LIBRARIES
find_package(Torch REQUIRED)
print_variable(TORCH_INCLUDE_DIRS)
print_variable(TORCH_LIBRARIES)
#
# CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#
# include_directories
include_directories( SYSTEM ${TORCH_INCLUDE_DIRS} )
include_directories( ${ad_tensor_SOURCE_DIR}/include )
#
# plugin_LIBRARIES, include_directories
set(plugin_LIBRARIES "")
if( include_plugin )
    #
    # Boost_INCLUDE_DIRS, Boost_LIBRARIES
    cmake_policy(SET CMP0167 NEW)
    find_package(Boost REQUIRED COMPONENTS filesystem system)
    print_variable(Boost_INCLUDE_DIRS)
    print_variable(Boost_LIBRARIES)
    #
    # include_directories
    include_directories( ${Boost_INCLUDE_DIRS} )
    include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
    #
    # plugin_LIBRARIES
    set(plugin_LIBRARIES "${Boost_LIBRARIES}")
endif()
#
# pkgconfig, src
add_subdirectory(pkgconfig)
add_subdirectory(src)
#
# examples, tests
if( include_gtest )
    find_package(GTest REQUIRED)
    add_subdirectory(examples)
    add_subdirectory(tests)
    add_subdirectory(benchmarks)
endif()
#
# install
install(TARGETS ad_tensor)
install(DIRECTORY include/ DESTINATION include)

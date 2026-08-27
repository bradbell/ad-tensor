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
}

Using CMake to Build, Test, and Install ad_tensor
#################################################

Command
*******
{xrst_code sh}
    cmake -S source_dir -B build_dir \
        -G generator \
        -D include_tests=(true|false) \
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

include_tests
=============
If include_tests is true (false), the following google test targets
will be included (will not be included) in the build:
examples, benchmarks, and tests.
This will require the google test package.

include_plugin
==============
If include_plugin is true (false), the ability to compile and link
generated source code will be included in the build.
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
The following command will run the tests:
{xrst_code sh}
    build_dir/examples/examples
    build_dir/benchmarks/benchmarks
    build_dir/tests/tests
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
#
# cmake_minimum_required
# version 3.12 is required for c++20
cmake_minimum_required(VERSION 3.12)
#
# project
project(ad_tensor VERSION "2026.8.26")
#
# CXX_STANDARD
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#
# TORCH_INCLUDE_DIRS, TORCH_CXX_FLAGS, TORCH_LIBRARIES
find_package(Torch REQUIRED)
message(STATUS "TORCH_INCLUDE_DIRS = ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
#
# CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#
# include_directories
include_directories( SYSTEM ${TORCH_INCLUDE_DIRS} )
include_directories( ${ad_tensor_SOURCE_DIR}/include )
#
# src
add_subdirectory(src)
#
# examples, tests
if( include_tests )
    find_package(GTest REQUIRED)
    add_subdirectory(examples)
    add_subdirectory(tests)
    add_subdirectory(benchmarks)
endif()
#
# install
install(TARGETS ad_tensor)
install(DIRECTORY include/ DESTINATION include)

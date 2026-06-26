# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
# cmake_minimum_required
cmake_minimum_required(VERSION 3.10)
#
# ad_tensor_SOURCE_DIR
project(ad_tensor VERSION "2026.6.26")
#
# TORCH_INCLUDE_DIRS, TORCH_CXX_FLAGS, TORCH_LIBRARIES
find_package(Torch REQUIRED)
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
endif()

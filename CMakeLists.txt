# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
# cmake_minimum_required
cmake_minimum_required(VERSION 3.10)
#
# ad_tensor_SOURCE_DIR
project(ad_tensor VERSION "2026.5.11")
#
# include_directories
include_directories( ${ad_tensor_SOURCE_DIR}/include )
#
# If debug build
if( include_tests )
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()

# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
# plugin_TESTS, plugin_LIBRARIES
set(plugin_TESTS "")
set(plugin_LIBRARIES "")
if( include_plugin )
    #
    # Boost_INCLUDE_DIRS, Boost_LIBRARIES
    cmake_policy(SET CMP0167 NEW)
    find_package(Boost REQUIRED COMPONENTS filesystem system)
    message( STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}" )
    message( STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}" )
    #
    # include_directories
    include_directories( ${Boost_INCLUDE_DIRS} )
    include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
    #
    # plugin_LIBRARIES, plugin_TESTS
    set(plugin_LIBRARIES "${Boost_LIBRARIES}")
    set(plugin_TESTS
        plugin/test_add_int.cpp
        plugin/test_add_tensor.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/plugin/build_lib.cpp
    )
    #
    add_subdirectory(plugin)
endif()
#
# BEGIN_SORT_THIS_LINE_PLUS_2
add_executable(tests
    ${plugin_TESTS}
    adfn/optimize.cpp
    adfn/reverse_der.cpp
    adten/matmul.cpp
    adten/solve.cpp
    dev/base_op.cpp
    dev/op_enum.cpp
    dev/rng_depend.cpp
    dev/tape.cpp
    dev/vec_sets.cpp
)
# END_SORT_THIS_LINE_MINUS_2
#
target_link_libraries(tests
    ad_tensor
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    ${plugin_LIBRARIES}
)
if( include_plugin )
    add_dependencies(tests plugin_lib)
endif()

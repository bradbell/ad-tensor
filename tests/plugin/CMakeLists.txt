# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# ----------------------------------------------------------------------------
#
# build_lib.cpp
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/build_lib.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/build_lib.cpp
    @ONLY
)
#
# plugin_lib
add_library(plugin_lib MODULE
    add_int.cpp
    add_tensor.cpp
)
target_link_libraries(plugin_lib PRIVATE ${Boost_LIBRARIES} )

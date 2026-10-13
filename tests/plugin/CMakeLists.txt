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
# get_plugin.hpp
# Once we require cmake >= 3.26 can use COPY_FILE instead of COPY
file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/get_plugin.hpp
    DESTINATION
    ${CMAKE_CURRENT_BINARY_DIR}
)
#
# plugin_lib
add_library(plugin_lib MODULE
    add_int.cpp
    sub_tensor.cpp
)
target_link_libraries(plugin_lib
    PRIVATE
    ${Boost_LIBRARIES}
    ${TORCH_LIBRARIES}
)

# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# ----------------------------------------------------------------------------
#
# plugin_lib
add_library(plugin_lib MODULE
    add_int.cpp
    add_tensor.cpp
)
target_link_libraries(plugin_lib PRIVATE ${Boost_LIBRARIES} )

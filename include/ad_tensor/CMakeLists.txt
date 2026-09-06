# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
if( include_plugin )
    set(include_plugin, 1)
else( )
    set(include_plugin, 0)
endif( )
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configure.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/configure.hpp
)

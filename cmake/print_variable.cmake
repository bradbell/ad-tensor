# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
# print_variable(variable_name)
#
# prints variable name = variable value
macro(print_variable variable_name)
    message(STATUS "${variable_name} = ${${variable_name}}" )
endmacro( )


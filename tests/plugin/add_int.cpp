// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <boost/config.hpp>
#include <boost/dll/alias.hpp>
#include <iostream>

// add_int
int add_int_plugin(int x, int y) {
    return x + y;
}

// Export the function with an alias named "add_int"
BOOST_DLL_ALIAS(add_int_plugin, add_int)

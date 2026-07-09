// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/to_string.hpp>
namespace ad_tensor {
    //
    // print_con
    void adfn_t::print_con(void) const
    {   //
        std::cout << "Begin " + get_name() + ".print_con\n";
        std::cout << to_string( m_con );
        std::cout << "End " + get_name() + ".print_con\n";
    }
}

// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/atom.hpp>
#include <ad_tensor/dev/atom.hpp>
namespace ad_tensor {
    //
    // make_atom
    size_t make_atom(std::unique_ptr<base_atom_t>& base_atom_ptr) {
        dev::atom_global_t& global = dev::atom_global_t::singleton();
        size_t atom_id             = global.store(base_atom_ptr);
        return atom_id;
    }
}

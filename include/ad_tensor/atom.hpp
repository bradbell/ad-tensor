#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/base_atom.hpp>
namespace ad_tensor {
    // BEGIN_MAKE_ATOM
    size_t make_atom(std::unique_ptr<base_atom_t>& base_atom_ptr);
    // END_MAKE_ATOM
}

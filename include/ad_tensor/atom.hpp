#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/base_atom.hpp>
#include <ad_tensor/adten.hpp>
namespace ad_tensor {
    size_t make_atom(std::unique_ptr<base_atom_t>& base_atom_ptr);
    vector<adten_t> call_atom(
        size_t                 atom_id   ,
        size_t                 call_info ,
        const vector<adten_t>& adomain
    );
}

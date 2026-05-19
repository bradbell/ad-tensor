#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
*/
#include <vector>
#include <ad_tensor/devel/agraph.hpp>
//
// BEGIN_ADFN_T
namespace ad_tensor { class adfn_t
// END_ADFN_T
}
private:
    std::vector<torch::Tensor> con_;
    agraph_t                   par_;
    agraph_t                   var_;
    size_t                     rng_index;
public:
    //
    // BEGIN_DEFAULT_CTOR
    adfn_t()
    // END_DEFAULT_CTOR
    : con_(), par_(), var_(), rmg_index(0)
    { }
    //
    // BEGIN_IS_EMPTY
    bool is_empty(void) const
    // END_IS_EMPTY
    { return con_.empty() && par_.is_empty() && var_.is_empty(); }
}; } }
//

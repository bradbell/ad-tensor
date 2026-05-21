#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
*/
#include <vector>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/devel/agraph.hpp>
//
namespace ad_tensor { class record_t; }
//
// BEGIN_ADFN_T
namespace ad_tensor { class adfn_t
// END_ADFN_T
    {
        friend class record_t;
    private:
        devel::agraph_t            par_;
        devel::agraph_t            var_;
        std::vector<at::Tensor>    con_;
        std::vector<size_t>        rng_index_;
        std::vector<ad_type_t>     rng_ad_type_;
    public:
        //
        // BEGIN_DEFAULT_CTOR
        adfn_t()
        // END_DEFAULT_CTOR
        : par_()
        , var_()
        , con_()
        , rng_index_()
        , rng_ad_type_()
        { }
        //
        // BEGIN_IS_EMPTY
        bool is_empty(void) const
        // END_IS_EMPTY
        { return
            par_.is_empty() &&
            var_.is_empty() &&
            con_.empty() &&
            rng_index_.empty() &&
            rng_ad_type_.empty();
        }
    };
}

// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
/*
{xrst_begin adfn_range usr}
{xrst_spell
    rng
}

Get The Range Tensors
#####################

Prototype
*********
{xrst_literal ,
    BEGIN_RANGE, END_RANGE
}

all_par
*******
is the value of all the parameters as a function of dom_par.

all_var
*******
is the value of all the variables as a function of dom_par and dom_var.

rng_vec
*******
is the value of the range vector as a function of dom_par and dom_var.
The returned references are only valid for as long as
all_par, all_var, and adfn do not change.

Example
*******
{xrst_literal ,
    examples/adfn/range.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adfn_range}
*/
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// BEGIN_RANGE
// rng_vec = adfn.range(all_par, all_var)
std::vector<at::Tensor> adfn_t::range(
    const std::vector<at::Tensor>& all_par ,
    const std::vector<at::Tensor>& all_var
) const
// END_RANGE
{   //
    assert( m_rng_index.size() == m_rng_ad_type.size() &&
        "adfn: m_rng_index and m_rng_ad_type have different sizes"
    );
    //
    // rng_vec
    std::vector<at::Tensor> rng_vec;
    for(size_t i = 0; i < m_rng_index.size(); ++i)
    {   size_t    index    = m_rng_index.at(i);
        ad_type_t ad_type  = m_rng_ad_type.at(i);
        switch(ad_type) {
            //
            case ad_type_t::constant:
            rng_vec.push_back( m_con.at(index) );
            break;
            //
            case ad_type_t::parameter:
            rng_vec.push_back( all_par.at(index) );
            break;
            //
            case ad_type_t::variable:
            rng_vec.push_back( all_var.at(index) );
            break;
            //
            default:
            assert( false && "adfn.range: "
                "a range type is not constant, parameter or variable"
            );
        }
    }
    return rng_vec;
}

} // END_NAMESPACE_AD_TENSOR

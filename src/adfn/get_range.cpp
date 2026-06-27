// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
/*
{xrst_begin adfn_get_range usr}

Get The Range Tensors
#####################

Prototype
*********
{xrst_literal ,
    BEGIN_RANGE, END_RANGE
}
This returns the range tensors for

    range = adfn(dom_par, dom_var)

TensorType
**********
This is either at::Tensor or :ref:`adten-name` .

par_all
*******
is the value of all the parameters as a function of dom_par.

var_all
*******
is the value of all the variables as a function of dom_par and dom_var.

range
*****
is the value of the range vector as function of dom_par and dom_var.

    range = adfn(dom_par, dom_var)

Example
*******
{xrst_literal ,
    examples/adfn/get_range.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adfn_get_range}
*/
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// BEGIN_RANGE
// rng_vec = adfn.get_range(par_all, var_all)
template <class TensorType>
vector<TensorType> adfn_t::get_range(
    const vector<TensorType>& var_all ,
    const vector<TensorType>& par_all
) const
// END_RANGE
{   //
    assert( m_rng_index.size() == m_rng_ad_type.size() &&
        "adfn: m_rng_index and m_rng_ad_type have different sizes"
    );
    //
    // rng_vec
    vector<TensorType> rng_vec;
    for(size_t i = 0; i < m_rng_index.size(); ++i)
    {   size_t    index    = m_rng_index[i];
        ad_type_t ad_type  = m_rng_ad_type[i];
        switch(ad_type) {
            //
            case ad_type_t::constant:
            rng_vec.push_back( m_con[index] );
            break;
            //
            case ad_type_t::parameter:
            rng_vec.push_back( par_all[index] );
            break;
            //
            case ad_type_t::variable:
            rng_vec.push_back( var_all[index] );
            break;
            //
            default:
            assert( false && "adfn.get_range: "
                "a range type is not constant, parameter or variable"
            );
        }
    }
    return rng_vec;
}
template vector<at::Tensor> adfn_t::get_range(
    const vector<at::Tensor>& par_all ,
    const vector<at::Tensor>& var_all
) const;

} // END_NAMESPACE_AD_TENSOR

// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
/*
{xrst_begin adfn_get_range usr}

Get The Range Tensors
#####################

Syntax
******
{xrst_code cpp}
range = adfn.get_range(par_all, var_all)
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adfn.hpp
    BEGIN_GET_RANGE, END_GET_RANGE
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

    range = adfn(dom_var, dom_par)

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
template <class TensorType>
vector<TensorType> adfn_t::get_range(
    const vector<TensorType>& var_all ,
    const vector<TensorType>& par_all
) const
{   //
    assert( m_rng_index.size() == m_rng_ad_type.size() &&
        "adfn: m_rng_index and m_rng_ad_type have different sizes"
    );
    //
    // range
    vector<TensorType> range;
    for(size_t i = 0; i < m_rng_index.size(); ++i)
    {   size_t    index    = m_rng_index[i];
        ad_type_t ad_type  = m_rng_ad_type[i];
        switch(ad_type) {
            //
            case ad_type_t::constant:
            range.push_back( m_con[index] );
            break;
            //
            case ad_type_t::parameter:
            range.push_back( par_all[index] );
            break;
            //
            case ad_type_t::variable:
            range.push_back( var_all[index] );
            break;
            //
            default:
            assert( false && "adfn.get_range: "
                "a range type is not constant, parameter or variable"
            );
        }
    }
    return range;
}
template vector<at::Tensor> adfn_t::get_range(
    const vector<at::Tensor>& par_all ,
    const vector<at::Tensor>& var_all
) const;

} // END_NAMESPACE_AD_TENSOR

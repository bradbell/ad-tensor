// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/options.hpp>
#include <ad_tensor/dev/base_op.hpp>
#include <ad_tensor/dev/derive_op.hpp>
/*
{xrst_begin adfn_forward_par usr}

Compute The Dependent Parameters
********************************

dom_par
*******
Is the value of the domain parameters for this calculation.

options
*******

trace
=====
The only implemented key is "trace" and its value values
are "true" and "false".
The default value for "trace" is "false".



{xrst_end adfn_forward_par}
*/
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// adfn_t::forward_par
std::vector<at::Tensor> adfn_t::forward_par(
    std::vector<at::Tensor>&& dom_par ,
    const options_t&          options
) const {
    //
    // cout
    using std::cout;
    //
    // trace
    bool trace = false;
    for( auto itr = options.begin(); itr != options.end(); ++itr)
    {   if( itr->first == "trace" )
        {   std::string value = itr->second;
            if( value == "true" )
                trace = true;
            else {
                assert( value == "false" && "forward_par: "
                    "trace is not true of false"
                );
            }
        } else {
            assert( false && "forward_par: invalid key in options" );
        }
    }
    // dom_par
    assert( dom_par.size() == m_par.n_dom_ && "forward_par: "
        "dom_par does not have the expected number of tensors"
    );
    //
    // n_op, n_all, empty
    size_t n_op      = m_par.m_op_seq.size();
    size_t n_all     = m_par.n_dom_ + n_op;
    at::Tensor empty = torch::empty( {0} );
    //
    // all_par
    std::vector<at::Tensor> all_par = std::move( dom_par );
    all_par.resize( n_all, empty );
    //
    if( trace ) {
        cout << "Begin tracing adfn::forward_par\n";
        for(size_t i = 0; i < m_con.size(); ++i) {
            cout << "constant[" << i << "] = " << m_con[i] << "\n";
        }
        for(size_t i = 0; i < m_par.n_dom_; ++i) {
            cout << "dom_par[" << i << "] = " << all_par[i] << "\n";
        }
    }
    //
    // all_par
    for(size_t op_index = 0; op_index < n_op; ++op_index) {
        //
        // base_op
        dev::op_enum_t op_enum = m_par.m_op_seq.at( op_index );
        const dev::base_op_t& base_op = dev::op_enum2base_op( op_enum );
        //
        // all_par
        base_op.forward_par(op_index, m_par, m_con, all_par);
        //
        if( trace) {
            cout << "all_par[" << op_index << "] = " << all_par.at(op_index);
            cout << ", op = " << base_op.op_string() << "\n";
        }
    }
    if( trace ) {
        cout << "End tracing adfn::forward_par\n";
    }
    return all_par;
}

} // END_NAMESPACE_AD_TENSOR

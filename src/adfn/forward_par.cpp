// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/options.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/dev/base_op.hpp>
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/dev/to_string.hpp>
#include <ad_tensor/dev/user_assert.hpp>
#include <ad_tensor/dev/get_option.hpp>
/*
{xrst_begin adfn_forward_par usr}

Compute The Dependent Parameters
################################

Prototype
*********
{xrst_literal ,
    BEGIN_FORWARD_PAR, END_FORWARD_PAR
}

dom_par
*******
Is the value of the domain parameters for this calculation.

options
*******
The possible key,value pairs ( see :ref:`options-name` ) are

.. csv-table::
    :header-rows: 1

    Key, Default, Possible other values
    "trace", "false", "true"

par_all
*******
A tensor is a parameter tensor if it depends on the tensors in dom_par
and does not depend on the tensors in dom_var .
The vector par_all contains all the parameter tensors that are calculated.
The vector dom_par is a sub-vector at the beginning of par_all.

Example
*******
{xrst_literal ,
    examples/adfn/forward_par.cpp
    BEGIN_CPP, END_CPP
}

{xrst_end adfn_forward_par}
*/
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// BEGIN_FORWARD_PAR
// par_all = adfn.forward_par(dom_par, options)
template <class TensorType>
ad_tensor::vector<TensorType> adfn_t::forward_par(
    const ad_tensor::vector<TensorType>& dom_par ,
    const options_t&                     options
) const
// END_FORWARD_PAR
{
    // cout
    using std::cout;
    using std::string;
    //
    // dom_par
# ifndef NDEBUg
    const vector< vector<int64_t> >&  shapes = m_par.m_dom_shapes;
    string msg = "forward_par: ";
    if( dom_par.size() != shapes.size() ) {
        msg += "dom_par.size() = " + std::to_string( dom_par.size() );
        msg += " and its size for this adfn is ";
        msg += std::to_string( shapes.size() );
        dev::user_assert( false , msg );
    }
    for(size_t i = 0; i < shapes.size(); ++i) {
        c10::IntArrayRef shape = shapes[i];
        if( ! dom_par[i].sizes().equals( shape ) ) {
            msg += "dom_par[" + std::to_string(i) + "] shape is ";
            msg += dev::to_string( dom_par[i].sizes() );
            msg += " and its shape for this adfn is ";
            msg += dev::to_string( shape );
            dev::user_assert( false , msg );
        }
    }
# endif
    //
    // trace
    string           key           = "trace";
    string           default_value = "false";
    std::set<string> other_values  = { "true" };
    string value = dev::get_option(options, key, default_value, other_values);
    bool trace = value == "true";
    if( trace ) {
        cout << "Begin tracing adfn::forward_par\n";
        for(size_t i = 0; i < m_con.size(); ++i) {
            string element = dev::to_string( m_con[i] );
            cout << "constant[" << i << "] = " << element << "\n";
        }
    }
    //
    // n_op, n_all, empty
    size_t n_op      = m_par.m_op_seq.size();
    TensorType empty = TensorType( torch::empty( {0} ) );
    //
    // par_all
    ad_tensor::vector<TensorType> par_all =  dom_par ;
    par_all.resize( n_op, empty );
    //
    // par_all
    for(size_t op_index = 0; op_index < n_op; ++op_index) {
        //
        // base_op
        dev::op_enum_t op_enum = m_par.m_op_seq[ op_index ];
        const dev::base_op_t<TensorType>& base_op =
            dev::op_enum2derive_op<TensorType>( op_enum );
        //
        // par_all
        base_op.forward_par(op_index, m_par, m_con, par_all);
        //
        if( trace) {
            string element = dev::to_string( par_all[op_index] );
            cout << "par_all[" << op_index << "] = " << element;
            cout << ", " << dev::to_string(op_enum)  << "(";
            size_t start = m_par.m_arg_start[op_index];
            size_t stop  = m_par.m_arg_start[op_index + 1];
            for(size_t i = start; i < stop; ++i) {
                cout << "[" << m_par.m_arg_value[i] << ",";
                cout << dev::to_string( m_par.m_arg_type[i] ) << "]";
            }
            cout << ")\n";
        }
    }
    if( trace ) {
        cout << "End tracing adfn::forward_par\n";
    }
    return par_all;
}
template ad_tensor::vector<adten_t> adfn_t::forward_par(
    const ad_tensor::vector<adten_t>&    dom_par ,
    const options_t&                     options
) const;
template ad_tensor::vector<at::Tensor> adfn_t::forward_par(
    const ad_tensor::vector<at::Tensor>& dom_par ,
    const options_t&                     options
) const;

} // END_NAMESPACE_AD_TENSOR

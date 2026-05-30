// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/get_option.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/options.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/dev/base_op.hpp>
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/dev/to_string.hpp>
#include <ad_tensor/dev/user_assert.hpp>
/*
{xrst_begin adfn_forward_var usr}

Compute The Dependent Variables
###############################

Prototype
*********
{xrst_literal ,
    BEGIN_FORWARD_VAR, END_FORWARD_VAR
}

all_par
*******
is the value of all the parameters for this function.
This is usually calculated by :ref:`adfn_forward_par-name` .
In the special case where dom_par is empty, all_par is also empty
and need not be computed by adfn::forward_par .

dom_var
*******
Is the value of the domain variables for this calculation.

options
*******
The possible key,value pairs ( see :ref:`options-name` ) are

.. csv-table::
    :header-rows: 1

    Key, Default, Possible other values
    "trace", "false", "true"

all_var
*******
A tensor is a variable tensor if it depends on the tensors in dom_var.
The vector all_var contains all the variable tensors that are calculated.
The vector dom_var is a sub-vector at the beginning of all_par.

Example
*******
{xrst_literal ,
    examples/adfn/forward_var.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adfn_forward_var}
*/
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// BEGIN_FORWARD_VAR
// all_var = adfn.forward_var(all_par, dom_var, options)
ad_tensor::vector<at::Tensor> adfn_t::forward_var(
    const ad_tensor::vector<at::Tensor>& all_par ,
    const ad_tensor::vector<at::Tensor>& dom_var ,
    const options_t&               options
) const
// END_FORWARD_VAR
{
    // cout
    using std::cout;
    using std::string;
    using ad_tensor::dev::to_string;
    //
    // dom_var
    dev::user_assert( dom_var.size() == m_var.m_n_dom ,
        "forward_var: dom_var does not have the expected number of tensors"
    );
    //
    // trace
    string           key           = "trace";
    string           default_value = "false";
    std::set<string> other_values  = { "true" };
    string value = dev::get_option(options, key, default_value, other_values);
    bool trace = value == "true";
    if( trace ) {
        cout << "Begin tracing adfn::forward_var\n";
        for(size_t i = 0; i < m_con.size(); ++i) {
            string element = to_string( m_con[i] );
            cout << "constant[" << i << "] = " << element << "\n";
        }
        for(size_t i = 0; i < all_par.size(); ++i) {
            string element = to_string( all_par[i] );
            cout << "all_par[" << i << "] = " << element << "\n";
        }
    }
    //
    // n_op, n_all, empty
    size_t n_op      = m_var.m_op_seq.size();
    at::Tensor empty = torch::empty( {0} );
    //
    // all_var
    ad_tensor::vector<at::Tensor> all_var =  dom_var ;
    all_var.resize( n_op, empty );
    //
    // all_var
    for(size_t op_index = 0; op_index < n_op; ++op_index) {
        //
        // base_op
        dev::op_enum_t op_enum = m_var.m_op_seq[ op_index ];
        const dev::base_op_t& base_op = dev::op_enum2derive_op( op_enum );
        //
        // all_var
        base_op.forward_var(op_index, m_var, m_con, all_par, all_var);
        //
        if( trace) {
            string element = to_string( all_var[op_index] );
            cout << "all_var[" << op_index << "] = " << element;
            cout << ", " << to_string(op_enum)  << "(";
            size_t start = m_var.m_arg_start[op_index];
            size_t stop  = m_var.m_arg_start[op_index + 1];
            for(size_t i = start; i < stop; ++i) {
                cout << "[" << m_var.m_arg_value[i] << ",";
                cout << to_string( m_var.m_arg_type[i] ) << "]";
            }
            cout << ")\n";
        }
    }
    if( trace ) {
        cout << "End tracing adfn::forward_var\n";
    }
    return all_var;
}

} // END_NAMESPACE_AD_TENSOR

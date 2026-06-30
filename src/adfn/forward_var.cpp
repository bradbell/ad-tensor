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
/*
{xrst_begin adfn_forward_var usr}

Compute The Dependent Variables
###############################

Syntax
******
{xrst_code cpp}
var_all = adfn.forward_var(dom_var, par_all)
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adfn.hpp
    BEGIN_FORWARD_VAR, END_FORWARD_VAR
}

TensorType
**********
This is either at::Tensor or :ref:`adten-name` .

dom_var
*******
Is the value of the domain variables for this calculation.

par_all
*******
is the value of all the parameters for this function.
This is usually calculated by :ref:`adfn_forward_par-name` .
In the special case where dom_par is empty, par_all is also empty
and need not be computed by adfn::forward_par .


trace
*****
if :ref:`adfn@trace` is true, this calculation will be traced.

var_all
*******
A tensor is a variable tensor if it depends on the tensors in dom_var.
The vector var_all contains all the variable tensors.
The vector dom_var is a sub-vector at the beginning of par_all.

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
template <class TensorType>
vector<TensorType> adfn_t::forward_var(
    const vector<TensorType>& dom_var ,
    const vector<TensorType>& par_all ) const
{
    //
    // using
    using std::cout;
    using std::string;
    using ad_tensor::dev::to_string;
    //
    // dom_var
# ifndef NDEBUG
    const vector< vector<int64_t> >&  shapes = m_var.m_dom_shapes;
    string msg = get_name() + ".forward_var: ";
    if( dom_var.size() != shapes.size() ) {
        msg += "dom_var.size() = " + std::to_string( dom_var.size() );
        msg += " and its size for this adfn is ";
        msg += std::to_string( shapes.size() );
        dev::user_assert( false , msg );
    }
    for(size_t i = 0; i < shapes.size(); ++i) {
        c10::IntArrayRef shape = shapes[i];
        if( ! dom_var[i].sizes().equals( shape ) ) {
            msg += "dom_var[" + std::to_string(i) + "] shape is ";
            msg += dev::to_string( dom_var[i].sizes() );
            msg += " and its shape for this adfn is ";
            msg += dev::to_string( shape );
            dev::user_assert( false , msg );
        }
    }
# endif
    //
    // trace
    bool trace = m_options.get_trace();
    if( trace ) {
        cout << "Begin tracing " + get_name() + ".forward_var\n";
    }
    //
    // n_op, n_all, empty
    size_t n_op      = m_var.m_op_seq.size();
    TensorType empty = TensorType( torch::empty( {0} ) );
    //
    // var_all
    vector<TensorType> var_all =  dom_var ;
    var_all.resize( n_op, empty );
    //
    // var_all
    for(size_t op_index = 0; op_index < n_op; ++op_index) {
        //
        // base_op
        dev::op_enum_t op_enum = m_var.m_op_seq[ op_index ];
        const dev::base_op_t<TensorType>& base_op =
            dev::op_enum2derive_op<TensorType>( op_enum );
        //
        // var_all
        base_op.forward_var(op_index, m_var, m_con, par_all, var_all);
        //
        if( trace) {
            string element = to_string( var_all[op_index] );
            cout << "var_all[" << op_index << "] = " << element;
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
        cout << "End tracing " + get_name() + ".forward_var\n";
    }
    return var_all;
}
template vector<adten_t> adfn_t::forward_var(
    const vector<adten_t>&    dom_var ,
    const vector<adten_t>&    par_all
) const;
template vector<at::Tensor> adfn_t::forward_var(
    const vector<at::Tensor>& dom_var ,
    const vector<at::Tensor>& par_al
) const;

} // END_NAMESPACE_AD_TENSOR

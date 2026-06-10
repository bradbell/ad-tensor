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
{xrst_begin adfn_reverse_der usr}
{xrst_spell
    rng
}

Compute Derivative of A Range Direction Summation
#################################################

Prototype
*********
{xrst_literal ,
    BEGIN_REVERSE_DER, END_REVERSE_DER
}

par_all
*******
is the value of all the parameters for this function.
This is usually calculated by :ref:`adfn_forward_par-name` .
In the special case where dom_par is empty, par_all is also empty
and need not be computed by adfn::forward_par .

var_all
*******
is the value of all the variables for this function.
This is usually calculated by :ref:`adfn_forward_var-name` .
Since derivatives are only computed with respect to domain variables,
it does not make sense for dom_var to be empty.

rng_der
*******
This is the range direction that the derivative is computed with respect to.


options
*******
Trace the reverse derivative calculations when options.get_trace() is true.

dom_der
*******
is the domain derivative of the range space direction; i.e.

    dom_der =  (d / d dom_var) sum[ rng_der * adfn(dom_par, dom_var ) ]


Example
*******
{xrst_literal ,
    examples/adfn/reverse_der.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adfn_reverse_der}
*/
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// BEGIN_REVERSE_DER
// dom_der = adfn.reverse_der(par_all, var_all, rng_der, options)
template <class TensorType>
ad_tensor::vector<TensorType> adfn_t::reverse_der(
    const ad_tensor::vector<TensorType>& par_all ,
    const ad_tensor::vector<TensorType>& var_all ,
    const ad_tensor::vector<TensorType>& rng_der ,
    const options_t&                     options
) const
// END_REVERSE_DER
{
    // cout
    using std::cout;
    using std::string;
    using ad_tensor::dev::to_string;
    //
    // rng_der
# ifndef NDEBUg
    const vector< vector<int64_t> >&  shapes = m_rng_shapes;
    string msg = "reverse_der: ";
    if( rng_der.size() != shapes.size() ) {
        msg += "rng_der.size() = " + std::to_string( rng_der.size() );
        msg += " and the range size for this adfn is ";
        msg += std::to_string( shapes.size() );
        dev::user_assert( false , msg );
    }
    for(size_t i = 0; i < shapes.size(); ++i) {
        c10::IntArrayRef shape = shapes[i];
        if( ! rng_der[i].sizes().equals( shape ) ) {
            msg += "rng_der[" + std::to_string(i) + "] shape is ";
            msg += dev::to_string( rng_der[i].sizes() );
            msg += " and the range shape for this index and adfn is ";
            msg += dev::to_string( shape );
            dev::user_assert( false , msg );
        }
    }
# endif
    //
    // trace
    bool trace = options.get_trace();
    //
    // n_op, n_all, empty
    size_t n_op      = m_var.m_op_seq.size();
    TensorType empty = TensorType( torch::empty( {0} ) );
    //
    // all_der
    ad_tensor::vector<TensorType> all_der( n_op, empty );
    for(size_t i = 0; i < m_rng_index.size(); ++i) {
        if( m_rng_ad_type[i] == ad_type_t::variable )  {
            all_der[ m_rng_index[i] ] = rng_der[i];
        }
    }
    //
    // n_dom_var
    size_t n_dom_var = m_var.m_dom_shapes.size();
    //
    // all_der
    size_t op_index = n_op;
    while( n_dom_var < op_index )
    {   --op_index;
        //
        // all_der[op_index]
        // Only propcess this operator if its result is connected to the range
        if( all_der[op_index].numel() != 0 ) {
            //
            // base_op
            dev::op_enum_t op_enum = m_var.m_op_seq[ op_index ];
            const dev::base_op_t<TensorType>& base_op =
            dev::op_enum2derive_op<TensorType>( op_enum );
            //
            // all_der
            base_op.reverse_der(
                op_index, m_var, m_con, par_all, var_all, all_der
            );
            //
            if( trace) {
                string element = to_string( all_der[op_index] );
                cout << "all_der[" << op_index << "] = " << element;
                cout << ", " << to_string(op_enum)  << "(";
                size_t start = m_var.m_arg_start[op_index];
                size_t stop  = m_var.m_arg_start[op_index + 1];
                for(size_t i = start; i < stop; ++i) {
                    cout << "[" << m_var.m_arg_value[i] << ",";
                    cout << to_string( m_var.m_arg_type[i] ) << "]";
                }
                cout << ")\n";
            }
        } else {
            // no longer need this memory so free it.
            all_der[op_index] = TensorType( torch::empty( {0} ) );
        }
    }
    //
    // dom_der
    ad_tensor::vector<TensorType> dom_der;
    for(size_t j = 0; j < n_dom_var; ++j) {
        if( all_der[j].numel() == 0 ) {
            c10::IntArrayRef shape = var_all[j].sizes();
            dom_der.push_back( TensorType( torch::zeros( shape ) ) );
        } else {
            dom_der.push_back( all_der[j] );
        }
    }
    if( trace ) {
        for(size_t j = 0; j < n_dom_var; ++j) {
            string element = to_string( dom_der[j] );
            cout << "dom_der[" << j << "] = " << element << "\n";
        }
        cout << "End tracing adfn::reverse_der\n";
    }
    return dom_der;
}
template ad_tensor::vector<adten_t> adfn_t::reverse_der(
    const ad_tensor::vector<adten_t>&    par_all ,
    const ad_tensor::vector<adten_t>&    var_all ,
    const ad_tensor::vector<adten_t>&    rng_der ,
    const options_t&                     options
) const;
template ad_tensor::vector<at::Tensor> adfn_t::reverse_der(
    const ad_tensor::vector<at::Tensor>& par_all ,
    const ad_tensor::vector<at::Tensor>& var_all ,
    const ad_tensor::vector<at::Tensor>& rng_der ,
    const options_t&                     options
) const;

} // END_NAMESPACE_AD_TENSOR
